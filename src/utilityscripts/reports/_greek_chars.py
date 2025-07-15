"""
Effectively a data file mapping greek characters to text inputs.
"""

GREEK_CHAR_MAP = {
    "alpha": ("α", "\\alpha"),  # noqa: RUF001
    "\\alpha": ("α", "\\alpha"),  # noqa: RUF001
    "beta": ("β", "\\beta"),
    "\\beta": ("β", "\\beta"),
    "gamma": ("γ", "\\gamma"),  # noqa: RUF001
    "\\gamma": ("γ", "\\gamma"),  # noqa: RUF001
    "delta": ("δ", "\\delta"),
    "\\delta": ("δ", "\\delta"),
    "epsilon": ("ε", "\\epsilon"),
    "\\epsilon": ("ε", "\\epsilon"),
    "zeta": ("ζ", "\\zeta"),
    "\\zeta": ("ζ", "\\zeta"),
    "eta": ("η", "\\eta"),
    "\\eta": ("η", "\\eta"),
    "theta": ("θ", "\\theta"),
    "\\theta": ("θ", "\\theta"),
    "iota": ("ι", "\\iota"),  # noqa: RUF001
    "\\iota": ("ι", "\\iota"),  # noqa: RUF001
    "kappa": ("κ", "\\kappa"),
    "\\kappa": ("κ", "\\kappa"),
    "lambda": ("λ", "\\lambda"),
    "\\lambda": ("λ", "\\lambda"),
    "mu": ("μ", "\\mu"),
    "\\mu": ("μ", "\\mu"),
    "nu": ("ν", "\\nu"),  # noqa: RUF001
    "\\nu": ("ν", "\\nu"),  # noqa: RUF001
    "xi": ("ξ", "\\xi"),
    "\\xi": ("ξ", "\\xi"),
    "omicron": ("ο", "\\omicron"),  # noqa: RUF001
    "\\omicron": ("ο", "\\omicron"),  # noqa: RUF001
    "pi": ("π", "\\pi"),
    "\\pi": ("π", "\\pi"),
    "rho": ("ρ", "\\rho"),  # noqa: RUF001
    "\\rho": ("ρ", "\\rho"),  # noqa: RUF001
    "sigma": ("σ", "\\sigma"),  # noqa: RUF001
    "\\sigma": ("σ", "\\sigma"),  # noqa: RUF001
    "tau": ("τ", "\\tau"),
    "\\tau": ("τ", "\\tau"),
    "upsilon": ("υ", "\\upsilon"),  # noqa: RUF001
    "\\upsilon": ("υ", "\\upsilon"),  # noqa: RUF001
    "phi": ("φ", "\\phi"),
    "\\phi": ("φ", "\\phi"),
    "chi": ("χ", "\\chi"),
    "\\chi": ("χ", "\\chi"),
    "psi": ("ψ", "\\psi"),
    "\\psi": ("ψ", "\\psi"),
    "omega": ("ω", "\\omega"),
    "\\omega": ("ω", "\\omega"),
    "Alpha": ("Α", "\\Alpha"),  # noqa: RUF001
    "ALPHA": ("Α", "\\Alpha"),  # noqa: RUF001
    "\\Alpha": ("Α", "\\Alpha"),  # noqa: RUF001
    "Beta": ("Β", "\\Beta"),  # noqa: RUF001
    "BETA": ("Β", "\\Beta"),  # noqa: RUF001
    "\\Beta": ("Β", "\\Beta"),  # noqa: RUF001
    "Gamma": ("Γ", "\\Gamma"),
    "GAMMA": ("Γ", "\\Gamma"),
    "\\Gamma": ("Γ", "\\Gamma"),
    "Delta": ("Δ", "\\Delta"),
    "DELTA": ("Δ", "\\Delta"),
    "\\Delta": ("Δ", "\\Delta"),
    "Epsilon": ("Ε", "\\Epsilon"),  # noqa: RUF001
    "EPSILON": ("Ε", "\\Epsilon"),  # noqa: RUF001
    "\\Epsilon": ("Ε", "\\Epsilon"),  # noqa: RUF001
    "Zeta": ("Ζ", "\\Zeta"),  # noqa: RUF001
    "ZETA": ("Ζ", "\\Zeta"),  # noqa: RUF001
    "\\Zeta": ("Ζ", "\\Zeta"),  # noqa: RUF001
    "Eta": ("Η", "\\Eta"),  # noqa: RUF001
    "ETA": ("Η", "\\Eta"),  # noqa: RUF001
    "\\Eta": ("Η", "\\Eta"),  # noqa: RUF001
    "Theta": ("Θ", "\\Theta"),
    "THETA": ("Θ", "\\Theta"),
    "\\Theta": ("Θ", "\\Theta"),
    "Iota": ("Ι", "\\Iota"),  # noqa: RUF001
    "IOTA": ("Ι", "\\Iota"),  # noqa: RUF001
    "\\Iota": ("Ι", "\\Iota"),  # noqa: RUF001
    "Kappa": ("Κ", "\\Kappa"),  # noqa: RUF001
    "KAPPA": ("Κ", "\\Kappa"),  # noqa: RUF001
    "\\Kappa": ("Κ", "\\Kappa"),  # noqa: RUF001
    "Lambda": ("Λ", "\\Lambda"),
    "LAMBDA": ("Λ", "\\Lambda"),
    "\\Lambda": ("Λ", "\\Lambda"),
    "Mu": ("Μ", "\\Mu"),  # noqa: RUF001
    "MU": ("Μ", "\\Mu"),  # noqa: RUF001
    "\\Mu": ("Μ", "\\Mu"),  # noqa: RUF001
    "Nu": ("Ν", "\\Nu"),  # noqa: RUF001
    "NU": ("Ν", "\\Nu"),  # noqa: RUF001
    "\\Nu": ("Ν", "\\Nu"),  # noqa: RUF001
    "Xi": ("Ξ", "\\Xi"),
    "XI": ("Ξ", "\\Xi"),
    "\\Xi": ("Ξ", "\\Xi"),
    "Omicron": ("Ο", "\\Omicron"),  # noqa: RUF001
    "OMICRON": ("Ο", "\\Omicron"),  # noqa: RUF001
    "\\Omicron": ("Ο", "\\Omicron"),  # noqa: RUF001
    "Pi": ("Π", "\\Pi"),
    "PI": ("Π", "\\Pi"),
    "\\Pi": ("Π", "\\Pi"),
    "Rho": ("Ρ", "\\Rho"),  # noqa: RUF001
    "RHO": ("Ρ", "\\Rho"),  # noqa: RUF001
    "\\Rho": ("Ρ", "\\Rho"),  # noqa: RUF001
    "Sigma": ("Σ", "\\Sigma"),
    "SIGMA": ("Σ", "\\Sigma"),
    "\\Sigma": ("Σ", "\\Sigma"),
    "Tau": ("Τ", "\\Tau"),  # noqa: RUF001
    "TAU": ("Τ", "\\Tau"),  # noqa: RUF001
    "\\Tau": ("Τ", "\\Tau"),  # noqa: RUF001
    "Upsilon": ("Υ", "\\Upsilon"),  # noqa: RUF001
    "UPSIGMA": ("Υ", "\\Upsilon"),  # noqa: RUF001
    "\\Upsilon": ("Υ", "\\Upsilon"),  # noqa: RUF001
    "Phi": ("Φ", "\\Phi"),
    "PHI": ("Φ", "\\Phi"),
    "\\Phi": ("Φ", "\\Phi"),
    "Chi": ("Χ", "\\Chi"),  # noqa: RUF001
    "CHI": ("Χ", "\\Chi"),  # noqa: RUF001
    "\\Chi": ("Χ", "\\Chi"),  # noqa: RUF001
    "Psi": ("Ψ", "\\Psi"),
    "PSI": ("Ψ", "\\Psi"),
    "\\Psi": ("Ψ", "\\Psi"),
    "Omega": ("Ω", "\\Omega"),
    "OMEGA": ("Ω", "\\Omega"),
    "\\Omega": ("Ω", "\\Omega"),
}
