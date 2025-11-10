"""
Python file to contain methods for working with AS5216.
"""

from enum import StrEnum



def phi_ms_tension(*, f_yf, f_uf):
    """
    Calculate the reduction factor for an anchor or channel in tension
    
    Parameters
    ----------
    f_yf :
        The yield strength of the fastener. In MPa.
    f_uf :
        The ultimate strength of the fastener. in MPa.
    """
    
    phi_base = 5 * f_yf / (6 * f_uf)
    
    return min(phi_base, 1 / 1.4)
    
def phi_ms_shear(*, f_yf, f_uf):
    """
    Calculate the reduction factor for an anchor or channel in shear
    
    Parameters
    ----------
    f_yf :
        The yield strength of the fastener. In MPa.
    f_uf :
        The ultimate strength of the fastener. in MPa.
    """
    
    f_ratio = f_yf / f_uf
    
    
    if f_uf <= 800 and f_ratio <= 0.8:
    
        return min(f_ratio, 0.8)
        
    return 0.8
    
def phi_ms_ca():
    """
    The capacity reduction factor for an anchor-channel connection.
    """
    
    return 1 / 1.8
    
def phi_ms_l():
    """
    The capacity reduction factor for an anchor-channel lip in local failure.
    """
    
    return 1 / 1.8

def phi_ms_l_x(phi_inst):
    """
    The capacity reduction factor for an anchor-channel lip in local failure in the x-direction.
    
    Parameters
    ----------
    phi_inst :
        A reduction factor as per Appendix A.
    """
    
    return phi_ms_l * phi_inst

def phi_ms_re():
    """
    The capacity reduction factor for supplementary reinforcement.
    """
    
    return 0.8

def phi_m_c(phi_inst):
    """
    The capacity reduction factor for concrete.
    
    Parameters
    ----------
    phi_inst :
        A reduction factor as per T3.2.4 or Appendix A
    """
    
    phi_c = 1 / 1.5
    
    return phi_c * phi_inst
    
def phi_m_sp(phi_inst):
    """
    The capacity reduction factor for concrete splitting.
    
    Parameters
    ----------
    phi_inst :
        A reduction factor as per T3.2.4 or Appendix A
    """
    
    returb phi_m_c(phi_inst = phi_inst)

def phi_m_p(phi_inst):
    """
    The capacity reduction factor for concrete pull out.
    
    Parameters
    ----------
    phi_inst :
        A reduction factor as per T3.2.4 or Appendix A
    """
    
    returb phi_m_c(phi_inst = phi_inst)

def Concrete(StrEnum):
    
    CRACKED = "cracked"
    UNCRACKED = "uncracked"


def cracked(*, sigma_l, sigma_r: float = 3.0, f_ct: float = 0.0):
    """
    Is the concrete considered to be cracked or uncracked?
    
    Notes
    -----
    - For 2D elements (slabs etc) this needs to be checked in both directions.
    
    Parameters
    ----------
    sigma_l : float
        The stress in the concrete due to loads. In MPa.
    sigma_r : float
        The stress in the concrete due to restraint, shrinkage etc. In MPa.
        Recommended to be at least 3.0MPa
    f_ct : float
        The characteristic tensile strength of the concrete.
        Can conservatively be taken to be 0MPa
    """
    
    sigma_total = sigma_l + sigma_r
    
    return Concrete.UNCRACKED if sigma_total <= f_ct else Concrete.CRACKED