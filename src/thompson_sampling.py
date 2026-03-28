"""
src/thompson_sampling.py
=========================
Thompson Sampling with Beta-Bernoulli conjugate model for
exploration-exploitation scoring of unvisited / cold-start grid cells.

Each grid cell maintains its own Beta(α, β) posterior.  New cells start
with a uniform prior (α=1, β=1).
"""

from scipy.stats import beta as beta_dist


class ThompsonSampler:
    """
    Thompson Sampler using Beta distributions for binary reward signals.

    Each grid cell is modelled as a Bernoulli bandit arm with unknown
    success probability θ.  The posterior is Beta(α, β), updated online
    as reward observations arrive.

    Parameters
    ----------
    grid_ids : list of str
        List of grid cell identifiers to initialise.
    """

    def __init__(self, grid_ids):
        try:
            if not grid_ids:
                raise ValueError("grid_ids must be a non-empty list.")

            self._posteriors = {}
            for gid in grid_ids:
                self._posteriors[str(gid)] = {"alpha": 1, "beta": 1}

            print(
                f"[thompson_sampling] Initialised Thompson Sampler with "
                f"{len(self._posteriors)} arms (uniform prior alpha=1, beta=1)"
            )

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.__init__] Failed: {exc}"
            ) from exc

    def _validate_grid_id(self, grid_id):
        """Ensure grid_id exists in the sampler."""
        gid = str(grid_id)
        if gid not in self._posteriors:
            raise KeyError(
                f"grid_id '{gid}' not found in Thompson Sampler. "
                f"Available arms: {len(self._posteriors)}"
            )
        return gid

    def sample(self, grid_id):
        """
        Draw a single sample from the Beta posterior of the given cell.

        Parameters
        ----------
        grid_id : str
            Grid cell identifier.

        Returns
        -------
        float
            A random draw from Beta(α, β) in [0, 1].
        """
        try:
            gid = self._validate_grid_id(grid_id)
            alpha = self._posteriors[gid]["alpha"]
            beta_val = self._posteriors[gid]["beta"]
            sampled_value = float(beta_dist.rvs(alpha, beta_val))
            return sampled_value

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.sample] "
                f"Failed for grid_id='{grid_id}': {exc}"
            ) from exc

    def update(self, grid_id, reward):
        """
        Update the Beta posterior with an observed binary reward.

        Parameters
        ----------
        grid_id : str
            Grid cell identifier.
        reward : int or float
            Binary reward: 1 (success / profitable) or 0 (failure).
        """
        try:
            gid = self._validate_grid_id(grid_id)

            if reward not in (0, 1, 0.0, 1.0):
                raise ValueError(
                    f"reward must be binary (0 or 1), got {reward}"
                )

            reward = int(reward)
            self._posteriors[gid]["alpha"] += reward
            self._posteriors[gid]["beta"] += (1 - reward)

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.update] "
                f"Failed for grid_id='{grid_id}', reward={reward}: {exc}"
            ) from exc

    def get_probability_estimate(self, grid_id):
        """
        Return the posterior mean as a point estimate of profitability.

        E[θ] = α / (α + β)

        Parameters
        ----------
        grid_id : str
            Grid cell identifier.

        Returns
        -------
        float
            Posterior mean probability in [0, 1].
        """
        try:
            gid = self._validate_grid_id(grid_id)
            alpha = self._posteriors[gid]["alpha"]
            beta_val = self._posteriors[gid]["beta"]
            estimate = alpha / (alpha + beta_val)
            return float(estimate)

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.get_probability_estimate] "
                f"Failed for grid_id='{grid_id}': {exc}"
            ) from exc

    def is_cold_start(self, grid_id):
        """
        Check whether a cell has never received a reward update.

        A cell is cold-start if its posterior is still the uniform prior
        (α=1, β=1).

        Parameters
        ----------
        grid_id : str
            Grid cell identifier.

        Returns
        -------
        bool
            True if the cell has never been updated.
        """
        try:
            gid = self._validate_grid_id(grid_id)
            alpha = self._posteriors[gid]["alpha"]
            beta_val = self._posteriors[gid]["beta"]
            return alpha == 1 and beta_val == 1

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.is_cold_start] "
                f"Failed for grid_id='{grid_id}': {exc}"
            ) from exc

    def get_all_estimates(self):
        """
        Return posterior mean estimates for all grid cells.

        Returns
        -------
        dict
            Mapping grid_id → posterior mean probability.
        """
        try:
            estimates = {}
            for gid, params in self._posteriors.items():
                alpha = params["alpha"]
                beta_val = params["beta"]
                estimates[gid] = alpha / (alpha + beta_val)
            return estimates

        except Exception as exc:
            raise RuntimeError(
                f"[thompson_sampling.ThompsonSampler.get_all_estimates] "
                f"Failed: {exc}"
            ) from exc

    def __len__(self):
        return len(self._posteriors)

    def __repr__(self):
        return (
            f"ThompsonSampler(n_arms={len(self._posteriors)})"
        )
