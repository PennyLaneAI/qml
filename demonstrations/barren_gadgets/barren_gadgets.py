import pennylane as qml
from pennylane import numpy as np


class PerturbativeGadgets:
    """ Class to generate the gadget Hamiltonian corresponding to a given
    computational hamiltonian according to the gadget construction derived
    by Faehrmann & Cichy 
    
    Args: 
        perturbation_factor (float) : parameter controlling the magnitude of the
                                      perturbation (aa pre-factor to \lambda_max)
    """
    def __init__(self, perturbation_factor=1):
        self.perturbation_factor = perturbation_factor
    
    def gadgetize(self, Hamiltonian, target_locality=3):
        """Generation of the perturbative gadget equivalent of the given 
        Hamiltonian according to the proceedure in Cichy, Fährmann et al.
        Args:
            Hamiltonian (qml.Hamiltonian)   : target Hamiltonian to decompose
                                              into more local terms
            target_locality (int > 2)       : desired locality of the resulting 
                                              gadget Hamiltonian
        Returns:
            Hgad (qml.Hamiltonian)          : gadget Hamiltonian
        """
        # checking for unaccounted for situations
        self.run_checks(Hamiltonian, target_locality)
        computational_qubits, computational_locality, computational_terms = self.get_params(Hamiltonian)
        
        # total qubit count, updated progressively when adding ancillaries
        total_qubits = computational_qubits
        #TODO: check proper convergence guarantee
        gap = 1
        perturbation_norm = np.sum(np.abs(Hamiltonian.coeffs)) \
                          + computational_terms * (computational_locality - 1)
        lambda_max = gap / (4 * perturbation_norm)
        l = self.perturbation_factor * lambda_max
        sign_correction = (-1)**(computational_locality % 2 + 1)
        # creating the gadget Hamiltonian
        coeffs_anc = []
        coeffs_pert = []
        obs_anc = []
        obs_pert = []
        ancillary_register_size = int(computational_locality / (target_locality - 2))
        for str_count, string in enumerate(Hamiltonian.ops):
            previous_total = total_qubits
            total_qubits += ancillary_register_size
            # Generating the ancillary part
            for anc_q in range(previous_total, total_qubits):
                coeffs_anc += [0.5, -0.5]
                obs_anc += [qml.Identity(anc_q), qml.PauliZ(anc_q)]
            # Generating the perturbative part
            for anc_q in range(ancillary_register_size):
                term = qml.PauliX(previous_total+anc_q) @ qml.PauliX(previous_total+(anc_q+1)%ancillary_register_size)
                term = qml.operation.Tensor(term, *string.non_identity_obs[
                    (target_locality-2)*anc_q:(target_locality-2)*(anc_q+1)])
                obs_pert.append(term)
            coeffs_pert += [l * sign_correction * Hamiltonian.coeffs[str_count]] \
                        + [l] * (ancillary_register_size - 1)
        coeffs = coeffs_anc + coeffs_pert
        obs = obs_anc + obs_pert
        Hgad = qml.Hamiltonian(coeffs, obs)
        return Hgad

    def get_params(self, Hamiltonian):
        """ retrieving the parameters n, k and r from the given Hamiltonian
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian from which to get the
                                            relevant parameters
        Returns:
            computational_qubits (int)    : total number of qubits acted upon by 
                                            the Hamiltonian
            computational_locality (int)  : maximum number of qubits acted upon
                                            by a single term of the Hamiltonian
            computational_terms (int)     : number of terms in the sum 
                                            composing the Hamiltonian
        """
        # checking how many qubits the Hamiltonian acts on
        computational_qubits = len(Hamiltonian.wires)
        # getting the number of terms in the Hamiltonian
        computational_terms = len(Hamiltonian.ops)
        # getting the locality, assuming all terms have the same
        computational_locality = max([len(Hamiltonian.ops[s].non_identity_obs) 
                                      for s in range(computational_terms)])
        return computational_qubits, computational_locality, computational_terms
    
    def run_checks(self, Hamiltonian, target_locality):
        """ method to check a few conditions for the correct application of 
        the methods
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian of interest
            target_locality (int > 2)     : desired locality of the resulting 
                                            gadget Hamiltonian
        Returns:
            None
        """
        computational_qubits, computational_locality, _ = self.get_params(Hamiltonian)
        computational_qubits = len(Hamiltonian.wires)
        if computational_qubits != Hamiltonian.wires[-1] + 1:
            raise Exception('The studied computational Hamiltonian is not acting on ' + 
                            'the first {} qubits. '.format(computational_qubits) + 
                            'Decomposition not implemented for this case')
        # Check for same string lengths
        localities=[]
        for string in Hamiltonian.ops:
            localities.append(len(string.non_identity_obs))
        if len(np.unique(localities)) > 1:
            raise Exception('The given Hamiltonian has terms with different locality.' +
                            ' Gadgetization not implemented for this case')
        # validity of the target locality given the computational locality
        if target_locality < 3:
            raise Exception('The target locality can not be smaller than 3')
        ancillary_register_size = computational_locality / (target_locality - 2)
        if int(ancillary_register_size) != ancillary_register_size:
            raise Exception('The locality of the Hamiltonian and the target' + 
                             ' locality are not compatible. The gadgetization' + 
                             ' with "unfull" ancillary registers is not' + 
                             ' supported yet. Please choose such that the' + 
                             ' computational locality is divisible by the' + 
                             ' target locality - 2')

    def zero_projector(self, Hamiltonian, target_locality=3):
        """Generation of a projector on the zero state |00...0>
        as a sum of projectors |0><0| on each qubit
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian) : Hamiltonian to be gadgetized
        Returns:
            projector (qml.Hamiltonian)   : projector that can be used as 
                                            an observable to measure
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-2))
        ancillary_qubits = r * ktilde
        coeffs = [1/ancillary_qubits] * ancillary_qubits
        obs = []
        for qubit in range(n_comp, n_comp+ancillary_qubits):
            zero_state = np.array([1, 0])
            zero_projector = qml.Hermitian(np.outer(zero_state, zero_state), 
                                           qubit)
            obs.append(zero_projector)
        projector = qml.Hamiltonian(coeffs, obs)
        return projector
    
    def all_zero_projector(self, Hamiltonian, target_locality=3):
        """Generation of a rank 1 projector on the zero state |00...0>
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian) : Hamiltonian to be gadgetized
        Returns:
            projector (qml.Hamiltonian)   : projector that can be used as 
                                            an observable to measure
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-2))
        ancillary_qubits = r * ktilde
        zero_state = np.zeros((2**ancillary_qubits))
        zero_state[0] = 1
        zero_projector = qml.Hermitian(np.outer(zero_state, zero_state), 
                                       range(n_comp, n_comp + ancillary_qubits, 1))
        projector = qml.Hamiltonian([1], [zero_projector])
        return projector


