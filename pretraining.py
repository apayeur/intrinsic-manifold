from noisy_linear_model import NoisyLinearNetwork
from linearized_model import LinearizedModel
from nonlinear_model import NonlinearDeterministicNetwork
import numpy as np
import copy
import os
from utils import build_data_container, update_data_container, matrix_angle


def main():
    outfig_format = 'png'

    # Parameters
    size = (6, 100, 2)              # (input size, recurrent size, output size)
    nb_readouts = 100
    noise = 0.e-2
    lr_init = 0.1 / size[1] #0.5 / size[1]
    seeds = np.arange(10, dtype=int)
    stopping_crit_pretraining = 1e-5
    exponents_W = [0.55, 1.]        # W_0 ~ N(0, 1/N^exponent_W)
    lr = {0.55: lr_init, 1.: lr_init * nb_readouts}  # {0.55: lr_init, 1.: lr_init * nb_readouts}
    activation_function = 'linear'

    do_record_data = True
    global_mean_input_is_zero = False

    for exponent_W in exponents_W:
        # Manage save and load folders
        tag = (f"pretraining-N{size[1]}-Nreadouts{nb_readouts}-activation{activation_function}-V1-expW{exponent_W}")  # identification of this experiment
        save_dir = f"data/egd/{tag}"
        save_dir_results = f"results/egd/{tag}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_results):
            os.makedirs(save_dir_results)

        # Data containers
        if do_record_data:
            data = {'pretraining': build_data_container()}
            representation_alignment = np.zeros(len(seeds))
            tangent_kernel_alignment = np.zeros((len(seeds), 4))
            delta_W_norm = np.zeros(len(seeds))

        for seed_id, seed in enumerate(seeds):
            print(f'\n|================================= Seed {seed} ===================================|')
            print('\n|----------------------------- Initial training -----------------------------|')
            if activation_function in ['tanh', 'relu']:
                net0 = NonlinearDeterministicNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                                     exponent_W=exponent_W,
                                                     global_mean_input_is_zero=global_mean_input_is_zero,
                                                     rng_seed=seed, activation_function=activation_function)
            elif activation_function == 'linear':
                net0 = NoisyLinearNetwork(network_size=size[1], nb_readouts=nb_readouts, nb_inputs=size[0],
                                          exponent_W=exponent_W, global_mean_input_is_zero=global_mean_input_is_zero,
                                          rng_seed=seed_id, noise=noise)
            else:
                raise ValueError("'activation_function' should be `linear`, `relu` or `tanh`")
            if do_record_data:
                NTK_0 = net0.neural_tangent_kernel()
                RSM_0 = net0.representation_similarity_matrix()
                W_0 = copy.copy(net0.W)

            net0.plot_output(outfile_name=f"{save_dir_results}/SampleBeforeInitialTraining_seed{seed}.{outfig_format}")

            data_pretraining = net0.train(lr=lr[exponent_W], stopping_crit=stopping_crit_pretraining, do_record_data=do_record_data)

            net0.plot_output(outfile_name=f"{save_dir_results}/SampleEndInitialTraining_seed{seed}.{outfig_format}")

            if do_record_data:
                NTK = net0.neural_tangent_kernel()
                RSM = net0.representation_similarity_matrix()

                representation_alignment[seed_id] = matrix_angle(RSM, RSM_0)
                tangent_kernel_alignment[seed_id] = matrix_angle(NTK, NTK_0)
                delta_W_norm[seed_id] = np.linalg.norm(net0.W - W_0)
                print("Tangent kernel alignment", tangent_kernel_alignment[seed_id])
                print("Representational alignment", representation_alignment[seed_id])
                print("Norm of weight change", delta_W_norm[seed_id])
                update_data_container(data_pretraining, data['pretraining'])

        if do_record_data:
            param_dict = {'size': size,
                          'nb_seeds': len(seeds),
                          'lr_init': lr_init}
            np.save(f"{save_dir}/params", param_dict)
            np.save(f"{save_dir}/data", data)
            np.save(f"{save_dir}/representation_alignment", representation_alignment)
            np.save(f"{save_dir}/tangent_kernel_alignment", tangent_kernel_alignment)
            np.save(f"{save_dir}/delta_W_norm", delta_W_norm)


if __name__ == '__main__':
    main()
