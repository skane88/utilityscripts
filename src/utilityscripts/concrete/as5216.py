"""
Python file to contain methods for working with AS5216.
"""

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