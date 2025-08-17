# utils/auto_imports.py
# Python 3.9 compatible

def inject_common(ns, *, verbose=False, plot_prefs=None):
    """
    Inject common aliases into the notebook namespace:
      np -> numpy, pd -> pandas, plt -> matplotlib.pyplot
      sp -> scipy,  sy -> sympy (pretty printing), qp -> qmcpy (optional)

    plot_prefs:
      True  -> apply Matplotlib rcParams (see below)
      False -> skip
      None  -> follow env var AUTO_PLOT_PREFS in {"1","true","yes"}
    """
    import os

    def _try(alias, mod, post=None):
        if alias in ns:
            return
        try:
            module = __import__(mod, fromlist=['*'])
            ns[alias] = module
            if post:
                try:
                    post(module)
                except Exception as e:
                    if verbose:
                        print(f"[auto_imports] post({alias}) error:", e)
        except Exception as e:
            if verbose:
                print(f"[auto_imports] skipped {alias} ({mod}): {e}")

    # Core aliases
    _try('np',  'numpy')
    _try('pd',  'pandas')
    _try('plt', 'matplotlib.pyplot')
    _try('sp',  'scipy')
    _try('sy',  'sympy', post=lambda sy: sy.init_printing(use_unicode=True))
    _try('qp',  'qmcpy')  # optional

    # Decide whether to apply plotting preferences
    if plot_prefs is None:
        plot_prefs = os.environ.get('AUTO_PLOT_PREFS', '0').lower() in ('1','true','yes')

    if plot_prefs and 'plt' in ns:
        plt = ns['plt']
        rc = plt.rcParams

        # Figure / text sizing
        rc['figure.figsize']     = (6.0, 4.0)
        rc['figure.dpi']         = 120
        rc['savefig.dpi']        = 120
        rc['font.size']          = 12
        rc['axes.titlesize']     = 'medium'
        rc['axes.labelsize']     = 'medium'
        rc['xtick.labelsize']    = 10
        rc['ytick.labelsize']    = 10
        rc['legend.fontsize']    = 10

        # Axes / grid style
        rc['axes.grid']          = True
        rc['grid.alpha']         = 0.25
        rc['grid.linestyle']     = '--'
        rc['axes.spines.top']    = False
        rc['axes.spines.right']  = False

        # Lines / markers
        rc['lines.linewidth']    = 2.0
        rc['lines.markersize']   = 4.0

        try:
            from utils.notebook_header import have_tex_toolchain
        except ImportError:
            have_tex_toolchain = lambda: shutil.which("latex") is not None  # fallback minimal check

        # Enable LaTeX text + macro preamble, but only if LaTeX toolchain is available
        if have_tex_toolchain():
            rc['text.usetex'] = True
            try:
                from utils.latex_macros import MATPLOTLIB_PREAMBLE
                rc['text.latex.preamble'] = MATPLOTLIB_PREAMBLE
            except Exception as e:
                if verbose:
                    print("[auto_imports] WARNING: latex_macros import failed:", e)
        else:
            if verbose:
                print("[auto_imports] WARNING: LaTeX toolchain not found; skipping usetex=True.")

        # LaTeX smoke test — verify usetex works by drawing a tiny figure
        fig = plt.figure()
        try:
            plt.text(0.5, 0.5, r"$\vx,\ \norm{2}{x}$")
            fig.canvas.draw()
        except Exception as e:
            if verbose:
                print("[auto_imports] WARNING: LaTeX smoke test failed:", e)
        finally:
            plt.close(fig)

    # Tiny post-check (quiet unless something is missing)
    missing = [a for a in ('np','pd','plt') if a not in ns]
    if verbose or missing:
        print(f"[auto_imports] Ready. Missing: {missing}" if missing else "[auto_imports] Ready.")