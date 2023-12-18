# Neural manifolds and learning regimes in neural-interface tasks
Simulation of a BMI center-out reaching task using a static Gaussian linear recurrent neural network. See preprint bioRxiv 2023.03.11.532146; doi: https://doi.org/10.1101/2023.03.11.532146.

# Dependencies
* Conda env file `analysis/environment.yml`contains the python dependencies.


# Folders in the repository
*`data/` : contains data from simulations

*`results/`: contains figure elements


# How to produce figures from the paper
* Clone branch `paper`, *not* `master`.

* Figure 1: Run `fig1.py` (produces . Then run `egd_fig_perturbation.py`

* Figure 2: Run `fig2.py`. Then run `edg_fig_loss.py` (panels A-B), `egd_fig_gradW.py` (C), and `edg_fig_total_delta_W.py` (D). You might have to change the name of the load directory in accordance with what you used as `tag` in `fig2.py`.    

* Figure 3: For panel A, run `fig3A.py`, then run `edg_max_eigvals_through_learning.py`. For panels B and C, run `fig3BC.py` and then run `egd_instability.py`.

* Figure 4: Run `fig4.py`.

* Figure 5: 

* Supplementary figures: No support is provided for generating these figures, but most of them are simple variations the main text figures.