import math
import logging

from src.config import CONVERGENCE_TOLERANCE, get_current_tau # Use getter for potentially calibrated tau

logger = logging.getLogger(__name__)

# --- Glicko-2 Core Mathematical Functions ---

def _g(phi):
    """The g function in Glicko-2, mapping RD to a scaling factor."""
    return 1 / math.sqrt(1 + 3 * (phi**2) / (math.pi**2))

def _E(mu, mu_opponent, phi_opponent):
    """The E function, calculating expected score against an opponent."""
    g_val = _g(phi_opponent)
    try:
        exponent = -g_val * (mu - mu_opponent)
        # Prevent overflow in exp
        if exponent > 700: return 1e-12 # Effectively 0
        if exponent < -700: return 1.0 - 1e-12 # Effectively 1
        return 1 / (1 + math.exp(exponent))
    except OverflowError:
        # If exponent is extremely large negative, result is 1. Extremely large positive, result is 0.
        return 1.0 if exponent < 0 else 1e-12

def _v(mu, mu_opponent, phi_opponent):
    """The v function, estimating variance of the match outcome."""
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    # Avoid division by zero or near-zero if E is very close to 0 or 1
    if E_val < 1e-9 or E_val > 1.0 - 1e-9:
        # Return a very small variance (high certainty) equivalent
        # This corresponds to a large inverse variance, effectively infinity in limits
        # Using a large number avoids numerical issues later.
        # The original paper doesn't explicitly state this handling, but it's practical.
        return 1e-12 # Represents near-infinite v_inv

    v_inv = (g_val**2) * E_val * (1 - E_val)
    if abs(v_inv) < 1e-12: # Avoid division by zero if v_inv is tiny
        return 1e12 # Effectively infinite v
    return 1 / v_inv

def _delta(mu, mu_opponent, phi_opponent, v, outcome):
    """The delta function, calculating the change in rating based on outcome."""
    g_val = _g(phi_opponent)
    E_val = _E(mu, mu_opponent, phi_opponent)
    # No need to cap v here, _v handles extreme E values leading to extreme v_inv
    return v * g_val * (outcome - E_val)

def _compute_new_volatility(phi, v, delta, sigma, tau=None):
    """Computes the new volatility (sigma') using the Illinois algorithm (Regula Falsi)."""
    if tau is None:
        tau = get_current_tau() # Get potentially calibrated tau

    a = math.log(sigma**2)
    delta_sq = delta**2
    phi_sq = phi**2

    # Handle edge case where v is practically infinite (v_inv is zero)
    # From _v, v_inv = (g_val**2) * E_val * (1 - E_val)
    # If E is 0 or 1, v_inv is 0, v becomes infinite (represented by 1e12 or similar)
    # If v is infinite, the update terms involving v dominate or become ill-defined.
    # Glickman's paper suggests convergence issues might arise.
    # A practical approach: if variance is infinite, volatility update is unreliable.
    if v > 1e11: # If v is extremely large (near infinite)
        logger.debug(f"Skipping volatility update due to extremely large v ({v})")
        return sigma # Return current sigma as update is ill-defined

    # Internal function for the iterative algorithm
    def f(x):
        exp_x = math.exp(x)
        term_A = phi_sq + v + exp_x
        # Avoid division by zero in the denominator
        if abs(term_A) < 1e-12:
             # Simplified form if denominator is zero (implies large negative x?)
             # This case might indicate numerical instability.
             logger.warning("Denominator near zero in volatility function f(x).")
             return -(x - a) / (tau**2 + 1e-12) # Regularization term dominates

        term1_num = exp_x * (delta_sq - phi_sq - v - exp_x)
        term1_den = 2 * (term_A**2)

        # Avoid division by zero for the derivative-like term
        if abs(term1_den) < 1e-12:
            logger.warning("Denominator (term1_den) near zero in volatility function f(x).")
            # Fallback or alternative calculation might be needed
            # For now, let regularization dominate
            return -(x - a) / (tau**2 + 1e-12)

        term1 = term1_num / term1_den
        term2 = (x - a) / (tau**2 + 1e-12) # Add epsilon to tau**2 for safety
        return term1 - term2

    # Initial bracketing step (A and B)
    A = a
    if delta_sq > phi_sq + v:
        try:
            # This case implies the performance was much better than expected
            B = math.log(delta_sq - phi_sq - v)
        except ValueError:
            # Argument to log is non-positive, should not happen if delta_sq > phi_sq + v
            # but handle defensively.
            logger.warning("ValueError during initial B calculation (log)")
            return sigma # Fallback: no update
    else:
        # Performance was worse than or equal to expected, search downwards for B
        k = 1
        max_k = 20 # Limit iterations to prevent infinite loops
        found_B = False
        while k <= max_k:
            try:
                f_val_at_B_candidate = f(a - k * tau)
                if f_val_at_B_candidate < 0:
                    found_B = True
                    break # Found a suitable B
            except (OverflowError, ValueError) as e:
                # Calculation failed, likely due to extreme values
                logger.warning(f"Error calculating f(a - k*tau) for k={k}: {e}")
                # Stop searching downwards, might use fallback or existing sigma
                k = max_k + 1 # Force exit
                break
            k += 1

        if not found_B:
            # Could not find a B where f(B) < 0 within reasonable steps
            # This might indicate slow convergence or issues with parameters.
            logger.debug(f"Volatility update: Could not find suitable B after {max_k} steps.")
            # Often, this means volatility shouldn't change much, or the step size (tau) is wrong.
            return sigma # Fallback: no update
        B = a - k * tau

    # Regula Falsi (Illinois variant) iteration
    try:
        fA = f(A)
        fB = f(B)
    except (OverflowError, ValueError) as e:
        logger.warning(f"Error calculating initial fA or fB in volatility update: {e}")
        return sigma # Fallback: no update

    # Ensure fA and fB have opposite signs for the root to be bracketed
    if fA * fB >= 0:
         # This shouldn't happen if bracketing was correct, indicates numerical issue or bad initial state.
         logger.warning(f"Volatility update: f(A) and f(B) do not bracket the root (fA={fA}, fB={fB}). Returning current sigma.")
         # Possible causes: initial sigma too small/large, extreme match results, tau value.
         return sigma

    side = 0 # Tracks which side was updated last for Illinois modification
    max_iter = 50 # Limit iterations
    iter_count = 0

    while abs(B - A) > CONVERGENCE_TOLERANCE and iter_count < max_iter:
        try:
            denom_regula = fB - fA
            # Avoid division by zero
            if abs(denom_regula) < 1e-12: denom_regula = 1e-12 * math.copysign(1, denom_regula)

            # Standard Regula Falsi step
            C = A + (A - B) * fA / denom_regula
            fC = f(C)

        except (OverflowError, ValueError) as e:
             logger.warning(f"Error during volatility iteration {iter_count}: {e}")
             # Iteration failed, return the best estimate so far (midpoint or A/B?)
             # Returning the current sigma is safer for now.
             return sigma # Fallback

        # Update the bracket [A, B]
        if fC * fB <= 0: # Root is in [C, B]
            A = B
            fA = fB
            # Illinois modification: reduce function value if stuck on one side
            if side == -1: fA /= 2
            side = 1 # Now updating B side
        else: # Root is in [A, C]
            # Illinois modification
            if side == 1: fB /= 2
            side = -1 # Now updating A side
            # Note: Original Illinois has `fB = fC` here, and `A=A`, `fA=fA`
            # Let's stick to the standard update logic for now
            # A = A remains A
            # fA = fA remains fA

        # Update B and fB regardless
        B = C
        fB = fC

        iter_count += 1

    if iter_count >= max_iter:
        logger.debug(f"Volatility update did not converge within {max_iter} iterations.")
        # Return the midpoint of the last bracket as an approximation
        # Or simply return the previous sigma if convergence failed badly.
        # Let's return the new estimate (exp(A/2) or exp(B/2)?) - A is the better bound here
        # Fallback to old sigma might be safer if convergence fails
        return sigma

    # Converged: return the new volatility (sigma')
    # The root found is for x = ln(sigma'^2), so sigma' = exp(root/2)
    # A should be the closer bound after the loop terminates
    return math.exp(A / 2) 