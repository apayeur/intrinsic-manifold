from toy_model_new import NonlinearDeterministicNetwork


class NoisyLinearNetwork(NonlinearDeterministicNetwork):
    def __init__(self, network_size=100, nb_inputs=6, exponent_W=0.55, exponent_V=1, noise=0.,
                 global_mean_input_is_zero=False, do_z_score=False, rng_seed=1):
        super().__init__(network_size=network_size, nb_inputs=nb_inputs,
                         exponent_W=exponent_W, exponent_V=exponent_V,
                         global_mean_input_is_zero=global_mean_input_is_zero,
                         do_z_score=do_z_score, rng_seed=rng_seed,
                         activation_function='linear')
        self.noise = noise
