from matplotlib import pyplot as plt
import numpy as np

class Rod1D:
    def __init__(self, n_comp, L, A=None, E=None, rho=None):
        self.n_comp = n_comp
        self.L = L
        self.A = A
        if self.A is not None:
            self.generate_cross_sections()
        self.x = self.generate_coordinates()
        if E is not None:
            assert(len(E)==self.n_comp)
            self.E = E
        else:
            self.generate_young_modulus()
        if rho is not None:
            assert(len(rho)==self.n_comp)
            self.rho = rho
        else:
            self.generate_density()

    def generate_coordinates(self):
        l_comp = [self.L/self.n_comp for _ in range(self.n_comp)]
        x = np.cumsum(l_comp)
        x = np.insert(x,0,0)

        return x
    
    def generate_young_modulus(self):
        self.E = np.ones(self.n_comp) 

    def generate_density(self):
        self.rho = np.ones(self.n_comp)

    def generate_cross_sections(self):
        self.cross_sections = np.ones(self.n_comp)*self.A

    def set_cross_sections_from_inverse(self,cs_inv):
        for i_comp in range(self.n_comp):
            self.cross_sections[i_comp] = 1./cs_inv[i_comp]

    def visualize(self, file_name=None, save_fig=False):
        fig, ax = plt.subplots()

        bottom = 0
        for i_comp in range(self.n_comp):
            length = self.x[i_comp+1] - self.x[i_comp]
            width = self.cross_sections[i_comp]
            ax.bar(0, length, width=width, bottom=bottom, align='center', label=str(i_comp),edgecolor='black', facecolor='gray')
            bottom += length

        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_fig:
            plt.savefig(file_name)