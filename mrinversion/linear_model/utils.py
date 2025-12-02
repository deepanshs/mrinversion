import numpy as np
from scipy.optimize import minimize

from mrinversion.linear_model import SmoothLassoCV


class CVMinimizer:
    def __init__(
        self,
        inverse_dimension,
        compressed_K,
        compressed_s,
        guess_neglogalpha,
        guess_negloglambda,
        sigma=0.0,
        folds=10,
        xygrid="full",
    ):
        self.inverse_dimension = inverse_dimension
        self.compressed_K = compressed_K
        self.compressed_s = compressed_s
        self.guess = np.array([guess_neglogalpha, guess_negloglambda])
        self.sigma = sigma
        self.folds = folds
        self.xygrid = xygrid

    @staticmethod
    def _cv_min_withprint(loghyperparameters, self):
        alpha = 10 ** (-loghyperparameters[0])
        lam = 10 ** (-loghyperparameters[1])

        s_lasso_fit = SmoothLassoCV(
            alphas=np.array([alpha]),
            lambdas=np.array([lam]),
            inverse_dimension=self.inverse_dimension,
            sigma=self.sigma,
            folds=self.folds,
            xygrid=self.xygrid,
        )

        s_lasso_fit.fit(self.compressed_K, self.compressed_s, cv_map_as_csdm=False)
        cv = np.log10(s_lasso_fit.cv_map[0][0])
        print(f"alpha: {-np.log10(alpha)}, lambda: {-np.log10(lam)}, cv: {cv}")
        return cv

    @staticmethod
    def _cv_min_noprint(loghyperparameters, self):
        alpha = 10 ** (-loghyperparameters[0])
        lam = 10 ** (-loghyperparameters[1])

        s_lasso_fit = SmoothLassoCV(
            alphas=np.array([alpha]),
            lambdas=np.array([lam]),
            inverse_dimension=self.inverse_dimension,
            sigma=self.sigma,
            folds=self.folds,
            xygrid=self.xygrid,
        )

        s_lasso_fit.fit(self.compressed_K, self.compressed_s, cv_map_as_csdm=False)
        cv = np.log10(s_lasso_fit.cv_map[0][0])
        return cv

    @staticmethod
    def _print_result(this_result, print_steps, prev_result=None):
        if prev_result:
            if this_result.fun < prev_result.fun:
                if this_result.success:
                    print(
                        "This minimization improved on the previous result and \n"
                        "completed the minimization in the iterations given. "
                    )
                else:
                    print(
                        "This minimization improved on the previous result, but did \n"
                        "not finish in the given number of iterations."
                    )
            else:
                if this_result.success:
                    print("This minimization minimized to a worse solution.")
                else:
                    print(
                        "This minimization did not improve on the previous result, \n"
                        "and did not finish in the given number of iterations."
                    )
            if print_steps:
                print(f"This minimization results: \n\n{this_result}")
        else:
            if this_result.success:
                print("This minimization completed in the iterations given. ")
            else:
                print("This minimization did not finish in given number of iterations.")
            if print_steps:
                print(f"This minimization results: \n\n{this_result}")

        print("--------------------------------------------------\n")

    def _minimize(
        self, guess, tol, maxiter, print_funcevals, print_steps, prev_result=None
    ):
        if print_funcevals:
            simplex_res = minimize(
                fun=self._cv_min_withprint,
                x0=guess,
                args=(self),
                tol=tol,
                # bounds=((2,10), (2,10)),
                options={"maxiter": maxiter},
                method="Nelder-Mead",
            )
        else:
            simplex_res = minimize(
                fun=self._cv_min_noprint,
                x0=guess,
                args=(self),
                tol=tol,
                # bounds=((2,10), (2,10)),
                options={"maxiter": maxiter},
                method="Nelder-Mead",
            )
        self._print_result(
            this_result=simplex_res, print_steps=print_steps, prev_result=prev_result
        )
        return simplex_res

    def mincv_close(
        self, tol=0.0002, maxiter=500, print_steps=True, print_funcevals=False
    ):
        simplex_res1 = self._minimize(
            guess=self.guess,
            tol=tol * 1.5,
            maxiter=maxiter,
            print_funcevals=print_funcevals,
            print_steps=print_steps,
        )

        if simplex_res1.success:
            simplex_res2 = self._minimize(
                guess=simplex_res1.x,
                tol=tol,
                maxiter=maxiter,
                print_funcevals=print_funcevals,
                print_steps=print_steps,
                prev_result=simplex_res1,
            )
            if simplex_res2.fun < simplex_res1.fun:
                return simplex_res2
            else:
                return simplex_res1
        else:
            return simplex_res1

    def mincv(
        self,
        tol=0.0002,
        maxiter=100,
        print_steps=True,
        print_funcevals=False,
        guess_bounds=[[2, 10], [2, 10]],
        guesses="all",
    ):
        '''options for guesses: "all", "cartesian", "diagonals"'''
        lowval_alpha = np.average([self.guess[0], guess_bounds[0][0]])
        highval_alpha = np.average([self.guess[0], guess_bounds[0][1]])
        lowval_lambda = np.average([self.guess[1], guess_bounds[1][0]])
        highval_lambda = np.average([self.guess[1], guess_bounds[1][1]])
        if guesses == "all":
            trythese = np.zeros((9, 2))
            trythese[0] = [self.guess[0], highval_lambda]
            trythese[1] = [highval_alpha, highval_lambda]
            trythese[2] = [highval_alpha, self.guess[1]]
            trythese[3] = [highval_alpha, lowval_lambda]
            trythese[4] = [self.guess[0], lowval_lambda]
            trythese[5] = [lowval_alpha, lowval_lambda]
            trythese[6] = [lowval_alpha, self.guess[1]]
            trythese[7] = [lowval_alpha, highval_lambda]
            trythese[8] = self.guess
        elif guesses == "cartesian":
            trythese = np.zeros((5, 2))
            trythese[0] = [self.guess[0], highval_lambda]
            trythese[1] = [highval_alpha, self.guess[1]]
            trythese[2] = [self.guess[0], lowval_lambda]
            trythese[3] = [lowval_alpha, self.guess[1]]
            trythese[4] = self.guess
        elif guesses == "diagonal":
            trythese = np.zeros((5, 2))
            trythese[0] = [highval_alpha, highval_lambda]
            trythese[1] = [highval_alpha, lowval_lambda]
            trythese[2] = [lowval_alpha, lowval_lambda]
            trythese[3] = [lowval_alpha, highval_lambda]
            trythese[4] = self.guess
        else:
            print('choose a valid choice for "guesses"')
            return None
        print(f"Starting points to try: {trythese}")
        first_step = [
            self._minimize(
                guess=thisguess,
                tol=tol * 5,
                maxiter=maxiter,
                print_funcevals=print_funcevals,
                print_steps=print_steps,
            )
            for thisguess in trythese
        ]
        print("-----------------------------------------------------------------")
        mins = np.asarray([step.fun for step in first_step])
        min_idx = np.where(mins == mins.min())[0][0]

        final_step = self._minimize(
            guess=first_step[min_idx].x,
            tol=tol,
            maxiter=maxiter,
            print_funcevals=print_funcevals,
            print_steps=False,
        )

        self._print_result(
            final_step, print_steps=True, prev_result=first_step[min_idx]
        )

        if final_step.fun > first_step[min_idx].fun:
            print("Second minimization did not improve.")
            return first_step[min_idx]
        else:
            return final_step
