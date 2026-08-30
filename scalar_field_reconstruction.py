
import numpy as np
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
import torch
from scipy import special
import os


DATA_FILE = "GlobalLandTemperaturesByCity.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
EARTH_RADIUS_METRES = 6378000
TEMPERATURE_DATE = "2003-07-01"
TEMPERATURE_COLUMNS = [
    "dt",
    "AverageTemperature",
    "Latitude",
    "Longitude",
    "City",
    "Country",
]


class Spherical_Harmonics:
    """Mathematical tools for spherical harmonic modelling."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L_cache = {}

    def spherical_harmonics_basis(self, l, m, theta, phi):
        """Return Y_l^m(theta, phi)."""
        return sph_harm_y(l, m, theta, phi)
    
    def convert_to_cartesian(self, theta, phi, r):
        """Convert spherical coordinates to Cartesian coordinates."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def associated_legendre(self, l, m, x):
        """Compute associated Legendre values using SciPy, then return to torch."""
        x_np = x.cpu().numpy()
        P_lm = special.lpmv(m, l, x_np)
        return torch.tensor(P_lm, dtype=torch.float64, device=self.device)
    
    def design_matrix(self, theta, phi, max_degree):
        """Build the regression design matrix."""
        A = np.zeros((len(theta), (max_degree + 1)**2), dtype=np.complex128)
        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                A[:, idx] = self.spherical_harmonics_basis(l, m, theta, phi)
                idx += 1
        return A

    def solve_lsq(self, max_degree, data, A, e, grad):
        """Solve the regularised least-squares system."""
        if e is None:
            e = 0.0
        L = self.construct_L(max_degree, grad)
        system = A.conj().T @ A + e * L
        rhs = A.conj().T @ data
        coefficients = np.linalg.solve(system, rhs)
        return coefficients  # Estimated spherical harmonic coefficients
    
    def design_matrix_gpu(self, theta, phi, max_degree):
        """GPU-friendly design matrix used during expensive grid-search steps."""
        num_points = len(theta)
        n_coeffs = (max_degree + 1)**2

        theta_np = theta.values if hasattr(theta, 'values') else np.array(theta)
        phi_np = phi.values if hasattr(phi, 'values') else np.array(phi)

        theta_torch = torch.tensor(theta_np, dtype=torch.float64, device=self.device)
        phi_torch = torch.tensor(phi_np, dtype=torch.float64, device=self.device)

        cos_theta = torch.cos(theta_torch)

        A = torch.zeros(
            (num_points, n_coeffs),
            dtype=torch.complex128,
            device=self.device,
        )

        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                P_lm = self.associated_legendre(l, m, cos_theta)
                
                norm = torch.sqrt(torch.tensor(
                    (2*l + 1) / (4 * np.pi) *
                    special.factorial(l - m) / special.factorial(l + m),
                    dtype=torch.float64, device=self.device
                ))

                phase = torch.exp(1j * m * phi_torch)

                A[:, idx] = norm * P_lm * phase
                idx += 1
        return A.cpu().numpy()
    
    def construct_L(self, max_degree, power):
        """Build the diagonal penalty matrix for higher-degree harmonics."""
        L = np.zeros(((max_degree + 1)**2, (max_degree + 1)**2), dtype=np.complex128)
        idx = 0
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                L[idx][idx] = (l * (l + 1))**power
                idx += 1
        return L
    
    def construct_L_cached(self, max_degree, power):
        """Reuse penalty matrices during grid search."""
        if (max_degree, power) in self.L_cache:
            return self.L_cache[(max_degree, power)]
        
        L = self.construct_L(max_degree, power)
        self.L_cache[(max_degree, power)] = L
        return L
    
    def gpu_solve(self, AtA, rhs, e, L):
        """Solve the regularised linear system on GPU."""
        return torch.linalg.solve(AtA + e * L, rhs)
    
    def plot_prediction_and_uncertainty(
        self,
        theta_grid,
        phi_grid,
        prediction,
        uncertainty,
        theta_cities,
        phi_cities,
        city_temp,
        r=EARTH_RADIUS_METRES,
    ):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Uncertainty surface
        xh, yh, zh = self.convert_to_cartesian(theta_grid, phi_grid, r)
        unc_norm = ((uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()))  # Normalise uncertainty
        colors = plt.cm.Reds(unc_norm)
        # Use stronger transparency where uncertainty is larger
        colors[..., -1] = unc_norm * 0.8
        ax.plot_surface(xh, yh, zh, facecolors=colors, rstride=1, cstride=1, shade=False)
        # City temperature points
        xc, yc, zc = self.convert_to_cartesian(theta_cities, phi_cities, r)
        scatter = ax.scatter(xc, yc, zc, c=city_temp, cmap='viridis', s=10)

        # Plot styling
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-60)
        ax.grid(True)  # Keep the 3D grid readable

        # Colourbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Temperature')
        ax.set_title("Temperature Prediction with Adaptive Geodesic Conformal Uncertainty Overlay")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/prediction_uncertainty_overlay.png", dpi=300, bbox_inches='tight')
        plt.show()


class SphericalHarmonicRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, harmonic_method, max_degree=None, reg_param=None, grad=None):
        self.harmonic_method = harmonic_method
        self.max_degree = max_degree
        self.reg_param = reg_param
        self.grad = grad
        self.coefficients = None

    def set_params(self, max_degree=None, reg_param=None, grad=None):
        """Set model parameters chosen by grid search."""
        if max_degree is not None:
            self.max_degree = max_degree
        if reg_param is not None:
            self.reg_param = reg_param
        if grad is not None:
            self.grad = grad
        return self

    def fit(self, X, y):
        """Fit the spherical harmonic coefficients."""
        A = self.harmonic_method.design_matrix(X[0], X[1], self.max_degree)
        self.coefficients = self.harmonic_method.solve_lsq(self.max_degree, y, A, self.reg_param, self.grad)
        return self

    def predict(self, X):
        """Predict temperatures at spherical locations."""
        A = self.harmonic_method.design_matrix(X[0], X[1], self.max_degree)
        return A @ self.coefficients


def process_temp_data():
    """Prepare raw city-temperature data for spherical harmonic fitting."""

    # Keep only one month before heavy processing
    chunks = []
    for chunk in pd.read_csv(
        DATA_FILE,
        delimiter=",",
        usecols=TEMPERATURE_COLUMNS,
        chunksize=50000,
    ):
        filtered = chunk[chunk["dt"] == TEMPERATURE_DATE]
        if not filtered.empty:
            chunks.append(filtered)
    data = pd.concat(chunks, ignore_index=True)

    data.rename(columns={"Latitude": "Theta", "Longitude": "Phi"}, inplace=True)
    data = data[["City", "Country", "AverageTemperature", "Theta", "Phi"]].dropna()

    def convert_latitude(lat):
        if lat[-1] == "N":
            return (90 - float(lat[:-1])) * np.pi / 180
        elif lat[-1] == "S":
            return (90 + float(lat[:-1])) * np.pi / 180
        else:
            raise ValueError("Invalid latitude format")
    
    data["Theta"] = np.array([convert_latitude(lat) for lat in data["Theta"]])
    
    def convert_longitude(long):
        if long[-1] == "W":
            return (360 - float(long[:-1])) * np.pi / 180
        elif long[-1] == "E":
            return (float(long[:-1])) * np.pi / 180
        else:
            raise ValueError("Invalid longitude format")
    
    data["Phi"] = np.array([convert_longitude(long) for long in data["Phi"]])

    print(f"Theta shape: {data['Theta'].shape}")
    print(f"Phi shape {data['Phi'].shape}")
    print(f"Temperatures: {data['AverageTemperature'].shape}")

    return data

def split_conformal_data(data, train_frac=0.6, validation_frac=0.2, cal_frac=0.1, random_seed=42):
    # Split data into train / validation / calibration / test sets for conformal prediction
    np.random.seed(random_seed)

    n = len(data)
    indices = np.random.permutation(n)

    n_train = int(train_frac * n)
    n_validation = int(validation_frac * n)
    n_cal = int(cal_frac * n)

    train_idx = indices[:n_train]
    validation_idx = indices[n_train:n_train + n_validation]
    cal_idx = indices[n_train + n_validation:n_train + n_validation + n_cal]
    test_idx = indices[n_train + n_validation + n_cal:]

    train_data = data.iloc[train_idx].reset_index(drop=True)
    validation_data = data.iloc[validation_idx].reset_index(drop=True)
    cal_data = data.iloc[cal_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validation_data)}")
    print(f"Calibration size: {len(cal_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, validation_data, cal_data, test_data

def split_data(data, num_folds=5, random_seed=42):
    """Create true K-fold train/validation splits for cross-validation."""

    train_data_list = []
    validation_data_list = []

    kfold = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed
    )

    for fold, (train_idx, validation_idx) in enumerate(
        kfold.split(data), start=1
    ):

        print(
            f"Fold {fold}: "
            f"Train size = {len(train_idx)}, "
            f"Validation size = {len(validation_idx)}, "
            f"Overlap = "
            f"{len(set(train_idx).intersection(set(validation_idx)))}"
        )

        train_data = data.iloc[train_idx].reset_index(drop=True)
        validation_data = data.iloc[validation_idx].reset_index(drop=True)

        train_data_list.append(train_data)
        validation_data_list.append(validation_data)

    return train_data_list, validation_data_list

def geodesic_distance(theta1, phi1, theta2, phi2):
    cos_angle = (
        np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        + np.cos(theta1) * np.cos(theta2)
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    return np.arccos(cos_angle)

def estimate_sigma_geodesic(
    theta_query,
    phi_query,
    theta_ref,
    phi_ref,
    residuals,
    k=20,
    eps=1e-6,
):
    sigma_values = []
    for tq, pq in zip(theta_query, phi_query):
        d = geodesic_distance(tq, pq, theta_ref, phi_ref)
        idx = np.argsort(d)[:k]  # Indices of k nearest neighbours
        sigma = np.mean(residuals[idx])  # Average neighbour residuals
        sigma = max(sigma, eps)  # Avoid division by zero
        sigma_values.append(sigma)
    return np.array(sigma_values)

def generate_cv_residuals(data, max_degree, reg_param, grad, num_folds=5):
    """Generate out-of-fold residuals for estimating local difficulty."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    harmonic_method = Spherical_Harmonics()
    residuals = []
    theta_vals = []
    phi_vals = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"CV Fold {fold + 1}/{num_folds}")
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        X_train = [train_data['Theta'].values, train_data['Phi'].values]
        y_train = train_data['AverageTemperature'].values
        X_val = [val_data['Theta'].values, val_data['Phi'].values]
        y_val = val_data['AverageTemperature'].values
        # Train model on this fold
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=reg_param,
            grad=grad,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val).real  # Predict validation fold
        fold_residuals = np.abs(y_val - preds)  # Absolute validation residuals
        residuals.extend(fold_residuals)
        theta_vals.extend(val_data['Theta'].values)
        phi_vals.extend(val_data['Phi'].values)
    return np.array(theta_vals), np.array(phi_vals), np.array(residuals)

# Search over lambda and gradient penalty to find the best model setting
def grid_search(data, max_degree, lambda_values, grad_values, base_model, num_folds):
    """Search over lambda and gradient penalty to find the best model setting."""
    harmonic_method = Spherical_Harmonics()
    train_data_list, validation_data_list = split_data(data, num_folds)

    print("Precomputing matrices...")

    A_training_matrices = []
    A_test_matrices = []
    train_values_list = []
    test_values_list = []

    for fold in range(num_folds):
        print(f"  Fold {fold + 1}/{num_folds}")
        train_data = train_data_list[fold]
        test_data = validation_data_list[fold]

        theta_train = train_data["Theta"].values
        phi_train = train_data["Phi"].values
        theta_test = test_data["Theta"].values
        phi_test = test_data["Phi"].values

        train_values = train_data["AverageTemperature"].values
        test_values = test_data["AverageTemperature"].values

        A_training_matrices.append(
            harmonic_method.design_matrix_gpu(theta_train, phi_train, max_degree)
        )
        A_test_matrices.append(
            harmonic_method.design_matrix_gpu(theta_test, phi_test, max_degree)
        )
        train_values_list.append(train_values)
        test_values_list.append(test_values)

    L_cache = {
        grad: harmonic_method.construct_L_cached(max_degree, grad)
        for grad in grad_values
    }
    device = harmonic_method.device
    fold_errors = []

    print("Testing parameter combinations...")

    grad_size = len(grad_values)
    lambda_size = len(lambda_values)
    total_combinations = grad_size * lambda_size * num_folds

    for fold in range(num_folds):
        A_train = torch.tensor(
            A_training_matrices[fold],
            dtype=torch.complex128,
            device=device,
        )
        A_test = torch.tensor(
            A_test_matrices[fold],
            dtype=torch.complex128,
            device=device,
        )
        train_values = torch.tensor(
            train_values_list[fold],
            dtype=torch.complex128,
            device=device,
        )
        test_values = torch.tensor(
            test_values_list[fold],
            dtype=torch.complex128,
            device=device,
        )

        ATA_train = A_train.conj().T @ A_train
        ATy = A_train.conj().T @ train_values
        comb_errors = np.zeros((grad_size, lambda_size))

        for j, reg_param in enumerate(lambda_values):
            for i, grad in enumerate(grad_values):
                if (j * grad_size + i) % 10 == 0:
                    completed = fold * lambda_size * grad_size + j * grad_size + i + 1
                    percentage = completed / total_combinations * 100
                    print(f"{percentage:.1f}%")

                L = torch.tensor(L_cache[grad], dtype=torch.complex128, device=device)
                coefficients = harmonic_method.gpu_solve(ATA_train, ATy, reg_param, L)
                predictions = A_test @ coefficients
                comb_errors[i, j] = torch.linalg.norm(predictions - test_values).item()

        fold_errors.append(comb_errors)

    mean_errors = np.mean(fold_errors, axis=0)
    print("\n===== Cross-validation errors =====")
    for i, grad in enumerate(grad_values):
        print(f"\nGradient = {grad}")
        for j, lam in enumerate(lambda_values):
            print(f"lambda = {lam:.2e}, error = {mean_errors[i, j]:.6f}")

    min_error = np.min(mean_errors)
    best_grad_index, best_lambda_index = np.unravel_index(
        np.argmin(mean_errors),
        mean_errors.shape,
    )
    log_errors = np.log10(mean_errors)
    log_min_error = np.log10(min_error)
    log_max_error = np.log10(np.max(mean_errors))
    norm_log_errors = (log_errors - log_min_error) / (log_max_error - log_min_error)

    ax = sns.heatmap(
        norm_log_errors,
        xticklabels=lambda_values,
        yticklabels=grad_values,
        cmap="viridis",
        fmt=".2f",
        cbar_kws={"label": "Error (normalized, log scale)"},
    )
    ax.set(xlabel="Regularisation Parameter", ylabel="k values")
    ax.set_title(f"Max Degree = {max_degree}")
    ax.add_patch(
        plt.Rectangle(
            (best_lambda_index, best_grad_index),
            1,
            1,
            fill=False,
            edgecolor="red",
            lw=3,
        )
    )

    plt.savefig(f"{PLOTS_DIR}/parameter_optimization_heatmap_L{max_degree}.png")
    plt.close()

    for grad_index, grad in enumerate(grad_values):
        plt.figure(figsize=(10, 6))
        lambda_errors = []

        for fold in range(num_folds):
            fold_lambda_errors = fold_errors[fold][grad_index, :]
            lambda_errors.append(fold_lambda_errors)
            plt.loglog(lambda_values, fold_lambda_errors, label=f"Fold {fold + 1}")

        plt.loglog(
            lambda_values,
            np.mean(lambda_errors, axis=0),
            label="Mean Error",
            linewidth=2,
            color="black",
        )
        plt.xlabel("Regularisation Parameter (log scale)")
        plt.ylabel("Error (log scale)")
        plt.title(f"Gradient penalised = {grad}")
        plt.legend()
        plt.savefig(f"{PLOTS_DIR}/parameter_optimization_L{max_degree}_grad_{grad}.png")
        plt.close()

    best_lambda = lambda_values[best_lambda_index]
    best_grad = grad_values[best_grad_index]
    print("\n--- GRID SEARCH RESULT ---")
    print(f"Best lambda: {best_lambda:.2e}")
    print(f"Best grad: {best_grad}")
    print(f"Minimum CV error: {min_error:.4f}")

    model = base_model
    model.set_params(max_degree=max_degree, reg_param=best_lambda, grad=best_grad)
    return model, best_lambda, best_grad

def save_model(model, name):
    """Save the trained model so reconstruction can run later."""
    with open(f"{MODELS_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_qhat(name="temp_qhat"):
    """Load the calibrated conformal threshold qhat."""
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        qhat = pickle.load(f)
    return qhat


def load_model(name):
    """Load the trained model for reconstruction mode."""
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def ensure_output_dirs():
    """Create output directories used by training and plotting."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_standard_coverage(model, cal_data, test_data, alpha=0.1):
    """Evaluate fixed-width split conformal intervals."""
    # Calibration residuals
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_preds = model.predict(X_cal).real
    cal_residuals = np.abs(y_cal - cal_preds)
    qhat = np.quantile(cal_residuals, 1 - alpha)
    # Test predictions
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_preds = model.predict(X_test).real
    lower = test_preds - qhat
    upper = test_preds + qhat
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)
    print("Coverage:", coverage)
    print("Average width:", width)
    return coverage, width


def evaluate_coverage(model, test_data, qhat, theta_cv, phi_cv, residuals, h=0.3):
    """Evaluate adaptive conformal coverage."""
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real

    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals,
        k=20,
    )

    lower = predictions - qhat * sigma_pred
    upper = predictions + qhat * sigma_pred
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    width = np.mean(upper - lower)
    print("Coverage:", coverage)
    print("Average interval width:", width)

    # High-uncertainty analysis
    q33 = np.quantile(sigma_pred, 0.33)
    q66 = np.quantile(sigma_pred, 0.66)
    low_mask = sigma_pred <= q33
    mid_mask = (sigma_pred > q33) & (sigma_pred <= q66)
    high_mask = sigma_pred > q66
    uncertainty_regions = [
        ("Low uncertainty", low_mask),
        ("Medium uncertainty", mid_mask),
        ("High uncertainty", high_mask),
    ]
    for name, mask in uncertainty_regions:
        region_cov = np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask]))
        region_width = np.mean(upper[mask] - lower[mask])
        print(f"\n{name}:")
        print("Coverage:", region_cov)
        print("Average width:", region_width)

    return coverage, width


def difficulty_diagnostic(model, test_data, theta_cv, phi_cv, residuals_cv):
    """Measure how well sigma(x) tracks absolute prediction residuals."""
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real
    true_residuals = np.abs(y_test - predictions)
    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    r = np.corrcoef(sigma_pred, true_residuals)[0, 1]
    print("\n--- DIFFICULTY DIAGNOSTIC ---")
    print("Correlation between sigma(x) and true residuals:", r)

    plt.figure(figsize=(6,5))
    plt.scatter(sigma_pred,true_residuals,alpha=0.5,s=20)
    # Linear regression trend
    m, b = np.polyfit(sigma_pred,true_residuals,1)
    x_line = np.linspace(sigma_pred.min(),sigma_pred.max(),100)
    plt.plot(x_line,m*x_line+b,'r',linewidth=2,label="Linear trend")
    plt.xlabel(r"Estimated difficulty $\hat{\sigma}(x)$")
    plt.ylabel("Absolute prediction residual")
    plt.title(f"Difficulty Estimator Validation\n"f"Correlation = {r:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/difficulty_validation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return r

def conditional_coverage_evaluation(
    model,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    n_bins=6,
):
    """Evaluate coverage after binning test points by estimated difficulty."""
    print("\n--- CONDITIONAL COVERAGE EVALUATION ---")
    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    predictions = model.predict(X_test).real
    sigma_pred = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )

    lower = predictions - qhat * sigma_pred
    upper = predictions + qhat * sigma_pred
    covered = (y_test >= lower) & (y_test <= upper)
    df = pd.DataFrame({'sigma': sigma_pred, 'covered': covered})
    df['difficulty_bin'] = pd.qcut(df['sigma'], q=n_bins, labels=False)

    results = []
    for b in range(n_bins):
        bin_df = df[df['difficulty_bin'] == b]
        bin_cov = bin_df['covered'].mean()
        results.append({
            'bin': b + 1,
            'count': len(bin_df),
            'avg_sigma': bin_df['sigma'].mean(),
            'coverage': bin_cov,
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    coverage_std = results_df['coverage'].std()
    worst_bin_coverage = results_df['coverage'].min()
    print("\nCoverage Std Across Bins:", coverage_std)
    print("Worst Bin Coverage:", worst_bin_coverage)
    return results_df


def plot_standard_vs_adaptive_diagnostic(
    model,
    cal_data,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    alpha=0.1,
    k_sigma=20,
):
    """Compare fixed-width standard CP with variable-width adaptive CP."""
    diagnostic_plot_path = f"{PLOTS_DIR}/standard_vs_adaptive_conformal_diagnostic.png"
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    standard_cal_residuals = np.abs(y_cal - cal_predictions)
    standard_qhat = np.quantile(standard_cal_residuals, 1 - alpha)

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    residuals = y_test - test_predictions
    abs_residuals = np.abs(residuals)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma,
    )

    standard_lower = test_predictions - standard_qhat
    standard_upper = test_predictions + standard_qhat
    standard_width = standard_upper - standard_lower
    standard_covered = (y_test >= standard_lower) & (y_test <= standard_upper)

    adaptive_lower = test_predictions - qhat * sigma_test
    adaptive_upper = test_predictions + qhat * sigma_test
    adaptive_width = adaptive_upper - adaptive_lower
    adaptive_covered = (y_test >= adaptive_lower) & (y_test <= adaptive_upper)

    lat = 90 - np.degrees(test_data['Theta'].values)
    lon = np.degrees(test_data['Phi'].values)
    lon = ((lon + 180) % 360) - 180

    diagnostic_df = pd.DataFrame({
        'city': test_data['City'].values,
        'country': test_data['Country'].values,
        'lat': lat,
        'lon': lon,
        'true_temp': y_test,
        'pred_temp': test_predictions,
        'residual': residuals,
        'abs_residual': abs_residuals,
        'sigma': sigma_test,
        'standard_width': standard_width,
        'adaptive_width': adaptive_width,
        'standard_covered': standard_covered,
        'adaptive_covered': adaptive_covered,
    })
    diagnostic_df['difficulty_bin'] = pd.qcut(
        diagnostic_df['sigma'],
        q=3,
        labels=['Low difficulty', 'Medium difficulty', 'High difficulty'],
    )

    summary_df = (
        diagnostic_df
        .groupby('difficulty_bin', observed=False)
        .agg(
            count=('city', 'size'),
            mean_sigma=('sigma', 'mean'),
            mean_abs_residual=('abs_residual', 'mean'),
            standard_coverage=('standard_covered', 'mean'),
            adaptive_coverage=('adaptive_covered', 'mean'),
            standard_width=('standard_width', 'mean'),
            adaptive_width=('adaptive_width', 'mean'),
        )
        .reset_index()
    )

    diagnostic_df.to_csv(
        f"{PLOTS_DIR}/standard_vs_adaptive_city_diagnostic.csv",
        index=False,
    )
    summary_df.to_csv(
        f"{PLOTS_DIR}/standard_vs_adaptive_difficulty_summary.csv",
        index=False,
    )

    width_min = min(np.nanmin(standard_width), np.nanmin(adaptive_width))
    width_max = max(np.nanmax(standard_width), np.nanmax(adaptive_width))
    size_min, size_max = 20, 180

    def width_to_size(width):
        if width_max == width_min:
            return np.full_like(width, (size_min + size_max) / 2)
        return size_min + (width - width_min) / (width_max - width_min) * (size_max - size_min)

    standard_sizes = width_to_size(standard_width)
    adaptive_sizes = width_to_size(adaptive_width)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    standard_scatter = axs[0, 0].scatter(
        lon,
        lat,
        c=standard_width,
        s=standard_sizes,
        cmap='Blues',
        vmin=width_min,
        vmax=width_max,
        edgecolors='black',
        linewidths=0.2,
        alpha=0.75,
    )
    axs[0, 0].set_title("(a) Standard CP: Fixed Interval Width")
    axs[0, 0].set_xlabel("Longitude")
    axs[0, 0].set_ylabel("Latitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.3)

    adaptive_scatter = axs[0, 1].scatter(
        lon,
        lat,
        c=adaptive_width,
        s=adaptive_sizes,
        cmap='Reds',
        vmin=width_min,
        vmax=width_max,
        edgecolors='black',
        linewidths=0.2,
        alpha=0.75,
    )
    axs[0, 1].set_title("(b) Adaptive CP: Variable Interval Width")
    axs[0, 1].set_xlabel("Longitude")
    axs[0, 1].set_ylabel("Latitude")
    axs[0, 1].grid(True, linestyle='--', alpha=0.3)

    plt.colorbar(standard_scatter, ax=axs[0, 0], label="Interval width")
    plt.colorbar(adaptive_scatter, ax=axs[0, 1], label="Interval width")

    axs[1, 0].scatter(sigma_test, abs_residuals, alpha=0.65, s=18)
    axs[1, 0].set_title("(c) Estimated Difficulty vs Absolute Residual")
    axs[1, 0].set_xlabel(r"Estimated difficulty $\hat{\sigma}(x)$")
    axs[1, 0].set_ylabel("Absolute residual")
    axs[1, 0].grid(True, linestyle='--', alpha=0.3)

    x = np.arange(len(summary_df))
    width = 0.35
    axs[1, 1].bar(
        x - width / 2,
        summary_df['standard_coverage'],
        width,
        label='Standard',
    )
    axs[1, 1].bar(
        x + width / 2,
        summary_df['adaptive_coverage'],
        width,
        label='Adaptive',
    )
    axs[1, 1].axhline(1 - alpha, color='black', linestyle='--', linewidth=1.2)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(summary_df['difficulty_bin'], rotation=15)
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].set_title("(d) Coverage by Difficulty Group")
    axs[1, 1].set_ylabel("Coverage")
    axs[1, 1].legend()
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(
        "Standard vs Adaptive Geodesic Conformal Diagnostics",
        fontsize=16,
        weight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        diagnostic_plot_path,
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
    plt.close(fig)

    print("\n--- STANDARD VS ADAPTIVE DIAGNOSTIC BY DIFFICULTY ---")
    print(summary_df.to_string(index=False, formatters={
        'mean_sigma': '{:.3f}'.format,
        'mean_abs_residual': '{:.3f}'.format,
        'standard_coverage': '{:.3f}'.format,
        'adaptive_coverage': '{:.3f}'.format,
        'standard_width': '{:.3f}'.format,
        'adaptive_width': '{:.3f}'.format,
    }))
    print("Saved diagnostic plot.")


def spherical_points_to_cartesian(theta, phi, radius=1.0):
    """Convert arrays of spherical coordinates to Cartesian points."""
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])


def geodesic_circle_points(theta, phi, angular_radius, n_points=160):
    """Create a geodesic circle on the unit sphere around one location."""
    center = spherical_points_to_cartesian(
        np.array([theta]),
        np.array([phi]),
        radius=1.0,
    )[0]

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(center, reference)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    u = np.cross(center, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(center, u)

    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = (
        np.cos(angular_radius) * center[None, :]
        + np.sin(angular_radius)
        * (
            np.cos(angles)[:, None] * u[None, :]
            + np.sin(angles)[:, None] * v[None, :]
        )
    )
    return circle


def plot_standard_vs_adaptive_geodesic_caps(
    model,
    cal_data,
    test_data,
    qhat,
    theta_cv,
    phi_cv,
    residuals_cv,
    alpha=0.1,
    k_sigma=20,
    max_points=10,
    output_path=f"{PLOTS_DIR}/standard_vs_adaptive_geodesic_caps.png",
):
    """Visualise fixed-size standard caps against variable-size adaptive caps."""
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    standard_qhat = np.quantile(np.abs(y_cal - cal_predictions), 1 - alpha)

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    abs_residuals = np.abs(y_test - test_predictions)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma,
    )

    standard_width = np.full(len(test_data), 2 * standard_qhat)
    adaptive_width = 2 * qhat * sigma_test

    standard_covered = abs_residuals <= standard_qhat
    adaptive_covered = abs_residuals <= qhat * sigma_test

    sorted_idx = np.argsort(sigma_test)
    if len(sorted_idx) > max_points:
        selected_idx = np.unique(
            np.linspace(0, len(sorted_idx) - 1, max_points).astype(int)
        )
        selected_idx = sorted_idx[selected_idx]
    else:
        selected_idx = sorted_idx

    selected_widths = adaptive_width[selected_idx]
    width_min = np.nanmin(selected_widths)
    width_max = np.nanmax(selected_widths)
    min_radius = 0.10
    max_radius = 0.45

    if width_max == width_min:
        adaptive_radii = np.full(len(selected_idx), (min_radius + max_radius) / 2)
    else:
        adaptive_radii = (
            min_radius
            + (selected_widths - width_min)
            / (width_max - width_min)
            * (max_radius - min_radius)
        )
    standard_radius = np.median(adaptive_radii)

    theta_test = test_data['Theta'].values
    phi_test = test_data['Phi'].values
    all_points = spherical_points_to_cartesian(theta_test, phi_test)
    selected_points = spherical_points_to_cartesian(
        theta_test[selected_idx],
        phi_test[selected_idx],
    )

    fig = plt.figure(figsize=(15, 7))
    axes = [
        fig.add_subplot(121, projection='3d', computed_zorder=False),
        fig.add_subplot(122, projection='3d', computed_zorder=False),
    ]
    titles = [
        "(a) Standard Geodesic CP: Fixed-Size Caps",
        "(b) Adaptive Geodesic CP: Variable-Size Caps",
    ]

    sphere_theta, sphere_phi = np.meshgrid(
        np.linspace(0, np.pi, 32),
        np.linspace(0, 2 * np.pi, 64),
    )
    sphere_x = np.sin(sphere_theta) * np.cos(sphere_phi)
    sphere_y = np.sin(sphere_theta) * np.sin(sphere_phi)
    sphere_z = np.cos(sphere_theta)

    for ax, title in zip(axes, titles):
        ax.plot_wireframe(
            sphere_x,
            sphere_y,
            sphere_z,
            color='lightgray',
            linewidth=0.4,
            alpha=0.25,
        )
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            color='gray',
            s=5,
            alpha=0.15,
        )
        ax.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            selected_points[:, 2],
            marker='^',
            color='seagreen',
            edgecolors='black',
            linewidths=0.4,
            s=45,
            label='Selected test cities',
            depthshade=False,
            zorder=10,
        )
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    for position, data_idx in enumerate(selected_idx):
        standard_circle = geodesic_circle_points(
            theta_test[data_idx],
            phi_test[data_idx],
            standard_radius,
        )
        adaptive_circle = geodesic_circle_points(
            theta_test[data_idx],
            phi_test[data_idx],
            adaptive_radii[position],
        )
        axes[0].plot(
            standard_circle[:, 0],
            standard_circle[:, 1],
            standard_circle[:, 2],
            color='steelblue',
            linewidth=1.3,
            alpha=0.75,
        )
        axes[1].plot(
            adaptive_circle[:, 0],
            adaptive_circle[:, 1],
            adaptive_circle[:, 2],
            color='steelblue',
            linewidth=1.3 + 2.2 * adaptive_radii[position] / max_radius,
            alpha=0.8,
        )

    for ax, covered in zip(axes, [standard_covered, adaptive_covered]):
        misses = selected_idx[~covered[selected_idx]]
        if len(misses) > 0:
            miss_points = spherical_points_to_cartesian(
                theta_test[misses],
                phi_test[misses],
            )
            ax.scatter(
                miss_points[:, 0],
                miss_points[:, 1],
                miss_points[:, 2],
                marker='*',
                color='gold',
                edgecolors='black',
                linewidths=0.4,
                s=75,
                label='Interval miss',
                depthshade=False,
                zorder=20,
            )
        ax.legend(loc='lower left', fontsize=8)

    fig.suptitle(
        "Standard vs Adaptive Geodesic Conformal Cap Diagnostic",
        fontsize=15,
        weight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print("Saved geodesic cap diagnostic.")


def lambda_grid_search_for_rmse_and_conformal(
    train_data,
    validation_data,
    max_degree,
    lambda_values,
    grad=2,
    alpha=0.1,
    num_folds=5,
    k_sigma=20,
    target_coverage=0.90,
):
    """Select lambda using RMSE and conformal validation criteria."""
    harmonic_method = Spherical_Harmonics()
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values
    X_validation = [validation_data['Theta'].values, validation_data['Phi'].values]
    y_validation = validation_data['AverageTemperature'].values
    results = []
    print(f"\n--- Lambda search for L = {max_degree} ---")
    for lam in lambda_values:
        print(f"Testing lambda = {lam:.2e}")
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=lam,
            grad=grad,
        )
        model.fit(X_train, y_train)
        # sigma(x) estimated using training data only
        theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
            train_data,
            max_degree=max_degree,
            reg_param=lam,
            grad=grad,
            num_folds=num_folds,
        )
        cv_rmse = np.sqrt(np.mean(residuals_cv ** 2))
        validation_predictions = model.predict(X_validation).real
        sigma_validation = estimate_sigma_geodesic(
            validation_data['Theta'].values,
            validation_data['Phi'].values,
            theta_cv,
            phi_cv,
            residuals_cv,
            k=k_sigma,
        )
        scores = np.abs(y_validation - validation_predictions) / np.maximum(
            sigma_validation,
            1e-6,
        )
        n = len(scores)
        q_level = np.ceil((1 - alpha) * (n + 1)) / n
        qhat = np.quantile(scores, q_level, method="higher")
        lower = validation_predictions - qhat * sigma_validation
        upper = validation_predictions + qhat * sigma_validation
        validation_coverage = np.mean((y_validation >= lower) & (y_validation <= upper))
        validation_width = np.mean(upper - lower)
        results.append({
            "lambda": lam,
            "cv_rmse": cv_rmse,
            "validation_coverage": validation_coverage,
            "validation_width": validation_width,
            "qhat": qhat,
        })
        print(
            f"  CV RMSE = {cv_rmse:.4f}, "
            f"validation coverage = {validation_coverage:.4f}, "
            f"width = {validation_width:.4f}"
        )
    results_df = pd.DataFrame(results)

    rmse_row = results_df.loc[results_df["cv_rmse"].idxmin()]
    rmse_lambda = rmse_row["lambda"]
    print(f"Best RMSE lambda for L = {max_degree}: {rmse_lambda:.2e}")
    print(f"Mean CV RMSE: {rmse_row['cv_rmse']:.4f}")

    valid = results_df[results_df["validation_coverage"] >= target_coverage]
    if not valid.empty:
        conformal_row = valid.loc[valid["validation_width"].idxmin()]
        print("\nSelected lambda: smallest width with coverage >= target")
    else:
        # fallback if no lambda reaches target coverage
        results_df["coverage_gap"] = np.abs(results_df["validation_coverage"] - target_coverage)
        conformal_row = results_df.sort_values(["coverage_gap", "validation_width"]).iloc[0]
        print("\nNo lambda reached target coverage; selected closest coverage with small width")
    conformal_lambda = conformal_row["lambda"]
    print(f"Best conformal lambda for L = {max_degree}: {conformal_lambda:.2e}")
    print(f"Validation coverage: {conformal_row['validation_coverage']:.4f}")
    print(f"Validation width: {conformal_row['validation_width']:.4f}")
    return rmse_lambda, conformal_lambda, results_df


def evaluate_degree_lambda(
    train_data,
    cal_data,
    test_data,
    L,
    best_lambda,
    best_grad,
    alpha,
    num_folds,
    k_sigma,
    selection_method,
):
    """Train and evaluate one degree/lambda combination."""
    harmonic_method = Spherical_Harmonics()
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values
    model = SphericalHarmonicRegressor(
        harmonic_method,
        max_degree=L,
        reg_param=best_lambda,
        grad=best_grad,
    )
    model.fit(X_train, y_train)

    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=L,
        reg_param=best_lambda,
        grad=best_grad,
        num_folds=num_folds
    )

    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    sigma_cal = estimate_sigma_geodesic(
        cal_data['Theta'].values,
        cal_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma
    )
    scores = np.abs(y_cal - cal_predictions) / np.maximum(sigma_cal, 1e-6)
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    qhat = np.quantile(scores, q_level, method="higher")

    X_test = [test_data['Theta'].values, test_data['Phi'].values]
    y_test = test_data['AverageTemperature'].values
    test_predictions = model.predict(X_test).real
    test_residuals = y_test - test_predictions
    abs_test_residuals = np.abs(test_residuals)

    standard_cal_residuals = np.abs(y_cal - cal_predictions)
    standard_qhat = np.quantile(standard_cal_residuals, 1 - alpha)
    standard_lower = test_predictions - standard_qhat
    standard_upper = test_predictions + standard_qhat
    standard_coverage = np.mean((y_test >= standard_lower) & (y_test <= standard_upper))
    standard_width = np.mean(standard_upper - standard_lower)

    sigma_test = estimate_sigma_geodesic(
        test_data['Theta'].values,
        test_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=k_sigma
    )
    adaptive_lower = test_predictions - qhat * sigma_test
    adaptive_upper = test_predictions + qhat * sigma_test
    adaptive_covered = ((y_test >= adaptive_lower) & (y_test <= adaptive_upper))
    adaptive_coverage = np.mean(adaptive_covered)
    adaptive_width = np.mean(adaptive_upper - adaptive_lower)
    correlation = np.corrcoef(sigma_test, abs_test_residuals)[0, 1]

    conditional_df = pd.DataFrame({"sigma": sigma_test, "covered": adaptive_covered})
    conditional_df["difficulty_bin"] = pd.qcut(
        conditional_df["sigma"],
        q=6,
        labels=False,
        duplicates="drop",
    )
    conditional_results = []
    for bin_id in sorted(conditional_df["difficulty_bin"].dropna().unique()):
        bin_df = conditional_df[conditional_df["difficulty_bin"] == bin_id]
        conditional_results.append({
            "bin": int(bin_id) + 1,
            "count": len(bin_df),
            "avg_sigma": bin_df["sigma"].mean(),
            "coverage": bin_df["covered"].mean(),
        })
    conditional_results_df = pd.DataFrame(conditional_results)

    return {
        "selection_method": selection_method,
        "L": L,
        "best_lambda": best_lambda,
        "grad": best_grad,
        "standard_coverage": standard_coverage,
        "adaptive_coverage": adaptive_coverage,
        "standard_width": standard_width,
        "adaptive_width": adaptive_width,
        "correlation": correlation,
        "conditional_coverage_std": conditional_results_df['coverage'].std(),
        "worst_bin_coverage": conditional_results_df['coverage'].min(),
        "rmse": np.sqrt(np.mean(test_residuals ** 2)),
        "mae": np.mean(abs_test_residuals),
    }

def run_degree_robustness_experiment(best_grad):
    ensure_output_dirs()

    data = process_temp_data()
    train_data, validation_data, cal_data, test_data = split_conformal_data(data)

    alpha = 0.1
    num_folds = 5
    grad_value = best_grad
    k_sigma = 20

    degrees = [8, 12, 16, 20, 24, 26, 28, 30]
    lambda_values = np.logspace(-8, 2, 21)
    print("DEGREES BEING TESTED:", degrees)
    rmse_results = []
    conformal_results = []

    for L in degrees:
        print(
            f"\n========== L = {L}: "
            "selecting lambda by RMSE and conformal interval ==========\n"
        )
        rmse_lambda, conformal_lambda, lambda_search_df = lambda_grid_search_for_rmse_and_conformal(
            train_data=train_data,
            validation_data=validation_data,
            max_degree=L,
            lambda_values=lambda_values,
            grad=grad_value,
            alpha=alpha,
            num_folds=num_folds,
            k_sigma=k_sigma,
            target_coverage=0.90,
        )
        lambda_search_df.to_csv(f"{PLOTS_DIR}/lambda_search_L{L}.csv", index=False)

        evaluated = {}
        for selection_method, selected_lambda in [
            ("RMSE", rmse_lambda),
            ("Conformal", conformal_lambda),
        ]:
            lambda_key = float(selected_lambda)
            if lambda_key not in evaluated:
                evaluated[lambda_key] = evaluate_degree_lambda(
                    train_data,
                    cal_data,
                    test_data,
                    L,
                    selected_lambda,
                    grad_value,
                    alpha,
                    num_folds,
                    k_sigma,
                    selection_method,
                )
            result = evaluated[lambda_key].copy()
            result["selection_method"] = selection_method
            if selection_method == "RMSE":
                rmse_results.append(result)
            else:
                conformal_results.append(result)

    rmse_results_df = pd.DataFrame(rmse_results)
    conformal_results_df = pd.DataFrame(conformal_results)
    comparison_df = pd.concat(
        [rmse_results_df, conformal_results_df],
        ignore_index=True,
    )

    print("\n========== DEGREE ROBUSTNESS RESULTS WITH RMSE-SELECTED LAMBDA ==========\n")
    print(rmse_results_df.to_string(index=False))
    print("\n========== DEGREE ROBUSTNESS RESULTS WITH CONFORMAL-SELECTED LAMBDA ==========\n")
    print(conformal_results_df.to_string(index=False))

    rmse_results_df.to_csv(f"{PLOTS_DIR}/degree_robustness_rmse_lambda_results.csv", index=False)
    conformal_results_df.to_csv(f"{PLOTS_DIR}/degree_robustness_conformal_lambda_results.csv", index=False)
    comparison_df.to_csv(f"{PLOTS_DIR}/degree_robustness_lambda_comparison_results.csv", index=False)

    plot_degree_robustness_results(
        rmse_results_df,
        title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees\nLambda selected by RMSE",
        output_path=f"{PLOTS_DIR}/degree_robustness_rmse_lambda_summary.png",
    )
    plot_degree_robustness_results(
        conformal_results_df,
        title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees\nLambda selected by conformal coverage with smaller intervals",
        output_path=f"{PLOTS_DIR}/degree_robustness_conformal_lambda_summary.png",
    )
    plot_lambda_selection_comparison(
        rmse_results_df,
        conformal_results_df,
        output_path=f"{PLOTS_DIR}/degree_robustness_lambda_comparison_summary.png",
    )

    return comparison_df

def plot_degree_robustness_results(
    results_df,
    title="Robustness of Adaptive Geodesic Conformal Prediction Across Spherical Harmonic Degrees",
    output_path=f"{PLOTS_DIR}/degree_robustness_summary.png",
):

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ======================================================
    # (a) Coverage
    # ======================================================
    axs[0, 0].plot(
        results_df["L"],
        results_df["standard_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Standard",
    )

    axs[0, 0].plot(
        results_df["L"],
        results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Adaptive",
    )

    axs[0, 0].axhline(
        y=0.90,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target (90%)",
    )

    axs[0, 0].set_title("(a) Coverage")
    axs[0, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 0].set_ylabel("Coverage")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # ======================================================
    # (b) Interval Width
    # ======================================================
    axs[0, 1].plot(
        results_df["L"],
        results_df["standard_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Standard",
    )

    axs[0, 1].plot(
        results_df["L"],
        results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Adaptive",
    )

    axs[0, 1].set_title("(b) Average Interval Width")
    axs[0, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 1].set_ylabel("Interval Width")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # ======================================================
    # (c) Correlation
    # ======================================================
    axs[1, 0].plot(
        results_df["L"],
        results_df["correlation"],
        "-o",
        linewidth=2,
        markersize=6,
    )

    axs[1, 0].set_title(r"(c) Correlation between $\hat{\sigma}(x)$ and Residual")
    axs[1, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].grid(True)

    # ======================================================
    # (d) RMSE
    # ======================================================
    axs[1, 1].plot(
        results_df["L"],
        results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
    )

    axs[1, 1].set_title("(d) RMSE")
    axs[1, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 1].set_ylabel("RMSE")
    axs[1, 1].grid(True)

    # ======================================================
    # Overall Figure
    # ======================================================
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def plot_lambda_selection_comparison(
    rmse_results_df,
    conformal_results_df,
    output_path=f"{PLOTS_DIR}/degree_robustness_lambda_comparison_summary.png",
):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(
        rmse_results_df["L"],
        rmse_results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[0, 0].plot(
        conformal_results_df["L"],
        conformal_results_df["adaptive_coverage"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[0, 0].axhline(
        y=0.90,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target (90%)",
    )
    axs[0, 0].set_title("(a) Adaptive Coverage")
    axs[0, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 0].set_ylabel("Coverage")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(
        rmse_results_df["L"],
        rmse_results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[0, 1].plot(
        conformal_results_df["L"],
        conformal_results_df["adaptive_width"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[0, 1].set_title("(b) Adaptive Interval Width")
    axs[0, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[0, 1].set_ylabel("Interval Width")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(
        rmse_results_df["L"],
        rmse_results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[1, 0].plot(
        conformal_results_df["L"],
        conformal_results_df["rmse"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[1, 0].set_title("(c) Test RMSE")
    axs[1, 0].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 0].set_ylabel("RMSE")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].semilogy(
        rmse_results_df["L"],
        rmse_results_df["best_lambda"],
        "-o",
        linewidth=2,
        markersize=6,
        label="RMSE-selected lambda",
    )
    axs[1, 1].semilogy(
        conformal_results_df["L"],
        conformal_results_df["best_lambda"],
        "-o",
        linewidth=2,
        markersize=6,
        label="Conformal-selected lambda",
    )
    axs[1, 1].set_title("(d) Selected Lambda")
    axs[1, 1].set_xlabel("Spherical Harmonic Degree (L)")
    axs[1, 1].set_ylabel("Lambda")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.suptitle(
        "Comparison of RMSE-Selected and Conformal-Selected Lambda",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_opened_sphere_map(
    theta_grid,
    phi_grid,
    temperature,
    uncertainty,
    theta_cities=None,
    phi_cities=None,
    city_values=None,
    city_uncertainty=None,
    city_names=None,
    country_names=None,
    label_uncertainty_percentile=95,
    max_labels=12,
    cut_longitude=0,
    name="opened_globe_uncertainty",
    title="Adaptive Geodesic Conformal Uncertainty Overlay",
):
    import matplotlib as mpl

    # Convert spherical coordinates
    lon_grid = np.degrees(phi_grid)
    lat_grid = 90 - np.degrees(theta_grid)
    lon_shifted = (lon_grid - cut_longitude) % 360
    # Sort latitude (north -> south) and longitude
    sort_lat_idx = np.argsort(lat_grid[:, 0])[::-1]
    sort_lon_idx = np.argsort(lon_shifted[0, :])
    unc_sorted = uncertainty[sort_lat_idx, :][:, sort_lon_idx]
    # Normalise uncertainty
    unc_low = np.nanpercentile(unc_sorted, 5)
    unc_high = np.nanpercentile(unc_sorted, 95)

    if unc_high == unc_low:
        unc_norm = np.zeros_like(unc_sorted)
    else:
        unc_norm = (unc_sorted - unc_low) / (unc_high - unc_low)
        unc_norm = np.clip(unc_norm, 0, 1)

    # Temperature colour limits
    if city_values is not None:
        vmin = np.nanmin(city_values)
        vmax = np.nanmax(city_values)
    else:
        vmin = np.nanmin(temperature)
        vmax = np.nanmax(temperature)

    # Figure
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_facecolor("white")

    # Red uncertainty overlay. Rows are latitude and columns are longitude,
    # so the array can be passed directly to imshow without transposing.
    red_rgba = plt.cm.Reds(unc_norm)
    red_rgba[..., -1] = 0.25 * unc_norm
    ax.imshow(
        red_rgba,
        extent=[cut_longitude, cut_longitude + 360, -90, 90],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )
    scatter = None

    # Plot cities
    if theta_cities is not None and phi_cities is not None:
        city_lat = 90 - np.degrees(theta_cities)
        city_lon = np.degrees(phi_cities)
        city_lon = (city_lon - cut_longitude) % 360 + cut_longitude
        scatter = ax.scatter(
            city_lon,
            city_lat,
            c=city_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=12,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )

        if city_names is not None and country_names is not None and city_uncertainty is not None:
            city_unc = np.asarray(city_uncertainty)
            threshold = np.nanpercentile(city_unc, label_uncertainty_percentile)
            high_unc_idx = np.where(city_unc >= threshold)[0]
            high_unc_idx = high_unc_idx[np.argsort(city_unc[high_unc_idx])[::-1]]
            high_unc_idx = high_unc_idx[:max_labels]
            for i in high_unc_idx:
                label = f"{city_names[i]}, {country_names[i]}"
                ax.text(
                    city_lon[i] + 1.0,
                    city_lat[i] + 0.6,
                    label,
                    fontsize=6,
                    color="black",
                    alpha=0.8,
                    zorder=4,
                )
    # Axes
    ax.set_xlim(cut_longitude, cut_longitude + 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)

    # Temperature colourbar
    if scatter is not None:
        temp_cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        temp_cbar.set_label("Temperature (°C)")

    # Uncertainty colourbar
    unc_mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=unc_low, vmax=unc_high),
        cmap=plt.cm.Reds,
    )
    unc_mappable.set_array([])
    unc_cbar = plt.colorbar(unc_mappable, ax=ax, pad=0.10)
    unc_cbar.set_label("Uncertainty")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def train_model():
    """Main entry point for training."""
    ensure_output_dirs()
    opt_choice = input('(1) Use grid search to optimise hyperparameters  (2) Manually enter parameters: ')
    data = process_temp_data()
    train_data, _, cal_data, test_data = split_conformal_data(data)

    print(train_data.head())
    print(cal_data.head())
    print(test_data.head())
    # Previous full-data input, kept as a reminder
    # Previous full-data target, kept as a reminder
    harmonic_method = Spherical_Harmonics()
    # Base model used by grid search
    base_model = SphericalHarmonicRegressor(
        harmonic_method,
        max_degree=1,
        reg_param=0.0,
        grad=0,
    )
    # Best gradient found by grid search
    best_grad = None
    # Use training set only for fitting
    X_train = [train_data['Theta'].values, train_data['Phi'].values]
    y_train = train_data['AverageTemperature'].values

    if opt_choice == '1':
        num_folds = int(input('Choose the number of folds to use: '))
        print("Define the parameter search space: ")
        n_lambda = int(input("Enter the number of regularisation parameters: "))
        max_lambda = int(input("Maximum regularisation parameter value (10^(?)): "))
        min_lambda = int(input("Minimum regularisation parameter value (10^(?)): "))
        lambda_values = np.logspace(int(min_lambda), int(max_lambda), int(n_lambda))

        max_grad = int(input("Enter the largest gradient to penalise: "))
        grad_values = np.arange(0, int(max_grad) + 1)

        max_degree = int(input("Enter the maximum degree of spherical harmonic: "))
        # Grid search now uses training data only
        # This avoids leaking calibration/test data into model selection
        model, best_lambda, best_grad = grid_search(
            data,
            max_degree,
            lambda_values,
            grad_values,
            base_model,
            num_folds,
        )
        print(f"\nBest lambda: {best_lambda}")
        print(f"Best gradient: {best_grad}")
        model.fit(X_train, y_train)

    elif opt_choice == '2':
        lambda_value = float(input("Enter the regularisation parameter: "))
        grad_value = int(input("Enter the gradient to penalise: "))
        max_degree = int(input("Enter the maximum degree of spherical harmonics: "))
        model = SphericalHarmonicRegressor(
            harmonic_method,
            max_degree=max_degree,
            reg_param=lambda_value,
            grad=grad_value,
        )
        model.fit(X_train, y_train)
        print(model.coefficients)
        # Use manually selected gradient
        best_grad = grad_value

    else:
        print('Invalid choice, please try again.')
        return train_model()

    alpha = 0.1
    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=model.max_degree,
        reg_param=model.reg_param,
        grad=model.grad,
        num_folds=5,
    )
    # Build residual field for local difficulty sigma(x)
    print("Geodesic residual field prepared.")
    # Compute adaptive conformal calibration scores
    X_cal = [cal_data['Theta'].values, cal_data['Phi'].values]
    y_cal = cal_data['AverageTemperature'].values
    cal_predictions = model.predict(X_cal).real
    sigma_pred = estimate_sigma_geodesic(
        cal_data['Theta'].values,
        cal_data['Phi'].values,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    scores = np.abs(y_cal - cal_predictions) / np.maximum(sigma_pred, 1e-6)
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    qhat = np.quantile(scores, q_level, method="higher")

    print("\n--- STANDARD CONFORMAL ---")
    evaluate_standard_coverage(model, cal_data, test_data, alpha)
    print("\n--- ADAPTIVE CONFORMAL ---")
    evaluate_coverage(model, test_data, qhat, theta_cv, phi_cv, residuals_cv)
    plot_standard_vs_adaptive_diagnostic(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
        alpha=alpha,
    )
    plot_standard_vs_adaptive_geodesic_caps(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
        alpha=alpha,
    )

    difficulty_diagnostic(model, test_data, theta_cv, phi_cv, residuals_cv)
    conditional_coverage_evaluation(model, test_data, qhat, theta_cv, phi_cv, residuals_cv)
    print("First 10 Calibration scores:")
    print(scores[:10])
    print("Conformal threshold qhat:", qhat)
    save_model(model, "temp_model")
    with open(f"{MODELS_DIR}/temp_qhat.pkl", "wb") as f:
        pickle.dump(qhat, f)
    # Return best gradient so main() can pass it
    # to the degree robustness experiment
    return model, best_grad

def reconstruct_field(n):
    """Reconstruct the scalar field on an n x n spherical grid."""
    ensure_output_dirs()
    high_uncertainty_percentile = 95
    high_uncertainty_max_labels = 20
    data = process_temp_data()
    train_data, _, cal_data, test_data = split_conformal_data(data)
    r = EARTH_RADIUS_METRES
    
    harmonic_method = Spherical_Harmonics()
    model = load_model("temp_model")
    qhat = load_qhat("temp_qhat")
    theta_cv, phi_cv, residuals_cv = generate_cv_residuals(
        train_data,
        max_degree=model.max_degree,
        reg_param=model.reg_param,
        grad=model.grad,
        num_folds=5,
    )
    plot_standard_vs_adaptive_diagnostic(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
    )
    plot_standard_vs_adaptive_geodesic_caps(
        model,
        cal_data,
        test_data,
        qhat,
        theta_cv,
        phi_cv,
        residuals_cv,
    )
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(0, np.pi, n),
        np.linspace(0, 2 * np.pi, n),
        indexing="ij",
    )
    theta = theta_grid.flatten()
    phi = phi_grid.flatten()
    
    A_torch = torch.tensor(
        harmonic_method.design_matrix_gpu(theta, phi, model.max_degree),
        dtype=torch.complex128,
        device=harmonic_method.device,
    )
    coefficients_torch = torch.tensor(
        model.coefficients,
        dtype=torch.complex128,
        device=harmonic_method.device,
    )

    reconstruction_torch = A_torch @ coefficients_torch
    reconstruction = reconstruction_torch.cpu().numpy()
    reconstruction = reconstruction.reshape(theta_grid.shape)

    sigma_pred = estimate_sigma_geodesic(theta, phi, theta_cv, phi_cv, residuals_cv, k=20)
    sigma_pred = sigma_pred.reshape(theta_grid.shape)
    lower = reconstruction.real - qhat * sigma_pred
    upper = reconstruction.real + qhat * sigma_pred
    uncertainty = upper - lower

    print("Conformal interval example at first grid point:")
    print("Prediction:", reconstruction.real[0, 0])
    print("Lower:", lower[0, 0])
    print("Upper:", upper[0, 0])

    theta_data = data['Theta'].values
    phi_data = data['Phi'].values
    X_data = [theta_data, phi_data]
    pred_data = model.predict(X_data).real

    # Compare observed and predicted city temperatures
    true_temp = data['AverageTemperature'].values
    residuals = true_temp - pred_data

    # Diagnostic metrics
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Bias: {bias:.3f}")

    # Residual distribution
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.savefig(f"{PLOTS_DIR}/residual_histogram.png", dpi=300)
    plt.show()

    # Observed vs reconstructed 3D comparison
    fig = plt.figure(figsize=(14, 6))
    vmin = min(true_temp.min(), pred_data.min())
    vmax = max(true_temp.max(), pred_data.max())

    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = harmonic_method.convert_to_cartesian(theta_data, phi_data, r)
    sc1 = ax1.scatter(x, y, z, c=true_temp, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("Observed City Temperatures")

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=pred_data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("Reconstructed Temperatures")
    plt.colorbar(sc1, ax=ax1, shrink=0.6)
    plt.colorbar(sc2, ax=ax2, shrink=0.6)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/observed_vs_reconstructed.png", dpi=300)
    plt.show()

    # Paper-style diagnostic plot
    latitudes = 90 - np.degrees(theta_data)  # Convert theta to latitude
    longitudes = np.degrees(phi_data)  # Convert longitude to degrees
    longitudes = ((longitudes + 180) % 360) - 180  # Convert longitude to [-180, 180]
    # Estimate adaptive uncertainty at city points
    sigma_city = estimate_sigma_geodesic(
        theta_data,
        phi_data,
        theta_cv,
        phi_cv,
        residuals_cv,
        k=20,
    )
    # Create dataframe
    plot_df = pd.DataFrame({
        'city': data['City'].values,
        'country': data['Country'].values,
        'theta': theta_data,
        'phi': phi_data,
        'lat': latitudes,
        'lon': longitudes,
        'true_temp': true_temp,
        'pred_temp': pred_data,
        'sigma': sigma_city,
    })
    # Sort by latitude for cleaner plotting
    plot_df = plot_df.sort_values('lat')
    plot_df['upper90'] = plot_df['pred_temp'] + qhat * plot_df['sigma']
    plot_df['lower90'] = plot_df['pred_temp'] - qhat * plot_df['sigma']

    # Longitude-sliced diagnostic plots
    lon_bins = [-180, -120, -60, 0, 60, 120, 180]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(6):
        lon_min = lon_bins[i]
        lon_max = lon_bins[i + 1]
        subset = plot_df[
            (plot_df['lon'] >= lon_min)
            & (plot_df['lon'] < lon_max)
        ].copy()
        subset = subset.sort_values('lat')
        ax = axes[i]

        ax.scatter(
            subset['lat'],
            subset['true_temp'],
            color='green',
            s=15,
            alpha=0.7,
            label='Observed',
        )
        ax.plot(
            subset['lat'],
            subset['pred_temp'],
            color='red',
            linewidth=2,
            label='Prediction',
        )
        ax.fill_between(
            subset['lat'],
            subset['lower90'],
            subset['upper90'],
            color='red',
            alpha=0.2,
            label='90% adaptive geodesic conformal interval',
        )

        outside_ci = subset[
            (subset['true_temp'] < subset['lower90'])
            | (subset['true_temp'] > subset['upper90'])
        ].copy()
        if not outside_ci.empty:
            outside_ci['ci_miss_size'] = np.maximum(
                outside_ci['lower90'] - outside_ci['true_temp'],
                outside_ci['true_temp'] - outside_ci['upper90'],
            )
            outside_ci = outside_ci.sort_values(
                'ci_miss_size',
                ascending=False,
            ).head(5)
            ax.scatter(
                outside_ci['lat'],
                outside_ci['true_temp'],
                color='gold',
                edgecolors='black',
                linewidths=0.5,
                s=35,
                alpha=0.90,
                zorder=5,
                label='Outside CI',
            )
            for _, row in outside_ci.iterrows():
                ax.text(
                    row['lat'] + 0.5,
                    row['true_temp'] + 0.4,
                    f"{row['city']}, {row['country']}",
                    fontsize=7,
                    color='black',
                    alpha=0.85,
                )

        ax.set_title(f'Longitude {lon_min} deg to {lon_max} deg')
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Temperature")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{PLOTS_DIR}/longitude_sliced_temperature.png", dpi=300)
    plt.show()

    plot_df['residual'] = plot_df['true_temp'] - plot_df['pred_temp']

    print("\n--- Highest Adaptive Geodesic Uncertainty Cities ---")
    high_uncertainty_threshold = np.nanpercentile(
        plot_df['sigma'],
        high_uncertainty_percentile,
    )
    high_uncertainty_cities = (
        plot_df[plot_df['sigma'] >= high_uncertainty_threshold]
        .sort_values('sigma', ascending=False)
        .head(high_uncertainty_max_labels)
        .copy()
    )
    high_uncertainty_cities['interval_width'] = (
        high_uncertainty_cities['upper90'] - high_uncertainty_cities['lower90']
    )

    density_radii_km = [250, 500, 1000]
    earth_radius_km = 6371
    all_theta = plot_df['theta'].values
    all_phi = plot_df['phi'].values

    for row_index, row in high_uncertainty_cities.iterrows():
        distances = geodesic_distance(
            row['theta'],
            row['phi'],
            all_theta,
            all_phi,
        )
        neighbour_distances = np.sort(distances[distances > 1e-12])

        for radius_km in density_radii_km:
            angular_radius = radius_km / earth_radius_km
            high_uncertainty_cities.loc[
                row_index,
                f'observations_within_{radius_km}km',
            ] = np.sum((distances <= angular_radius) & (distances > 1e-12))

        if len(neighbour_distances) >= 20:
            distance_to_20th = neighbour_distances[19] * earth_radius_km
        else:
            distance_to_20th = np.nan
        high_uncertainty_cities.loc[
            row_index,
            'distance_to_20th_neighbour_km',
        ] = distance_to_20th

    high_uncertainty_columns = [
        'city',
        'country',
        'lat',
        'lon',
        'true_temp',
        'pred_temp',
        'residual',
        'sigma',
        'lower90',
        'upper90',
        'interval_width',
        'observations_within_250km',
        'observations_within_500km',
        'observations_within_1000km',
        'distance_to_20th_neighbour_km',
    ]
    print(
        high_uncertainty_cities[high_uncertainty_columns]
        .to_string(index=False, formatters={
            'lat': '{:.2f}'.format,
            'lon': '{:.2f}'.format,
            'true_temp': '{:.2f}'.format,
            'pred_temp': '{:.2f}'.format,
            'residual': '{:.2f}'.format,
            'sigma': '{:.2f}'.format,
            'lower90': '{:.2f}'.format,
            'upper90': '{:.2f}'.format,
            'interval_width': '{:.2f}'.format,
            'observations_within_250km': '{:.0f}'.format,
            'observations_within_500km': '{:.0f}'.format,
            'observations_within_1000km': '{:.0f}'.format,
            'distance_to_20th_neighbour_km': '{:.1f}'.format,
        })
    )
    high_uncertainty_cities[high_uncertainty_columns].to_csv(
        f"{PLOTS_DIR}/high_uncertainty_cities_top5pct_top{high_uncertainty_max_labels}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.axis('off')
    table_df = high_uncertainty_cities[high_uncertainty_columns].copy()
    numeric_columns = [
        'lat',
        'lon',
        'true_temp',
        'pred_temp',
        'residual',
        'sigma',
        'lower90',
        'upper90',
        'interval_width',
        'observations_within_250km',
        'observations_within_500km',
        'observations_within_1000km',
        'distance_to_20th_neighbour_km',
    ]
    for column in numeric_columns:
        table_df[column] = table_df[column].map(lambda value: f"{value:.2f}")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.45)
    ax.set_title(
        "Highest-Uncertainty Cities Labelled on the Map",
        fontsize=12,
        weight='bold',
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_DIR}/high_uncertainty_cities_top5pct_top{high_uncertainty_max_labels}.png",
        dpi=300,
        bbox_inches='tight',
    )
    plt.close(fig)

    shared_temp_vmin = min(np.nanmin(pred_data), np.nanmin(true_temp))
    shared_temp_vmax = max(np.nanmax(pred_data), np.nanmax(true_temp))
    print(f"Shared opened-globe temperature scale: {shared_temp_vmin:.2f} to {shared_temp_vmax:.2f} deg C")

    # Geographic residual diagnostic plot
    plt.figure(figsize=(12, 6))
    sc = plt.scatter(
        plot_df['lon'],
        plot_df['lat'],
        c=plot_df['residual'],
        cmap='coolwarm',
        s=40,
    )
    plt.colorbar(sc, label='Residual')
    top_errors = plot_df.reindex(
        plot_df['residual'].abs().sort_values(ascending=False).index
    ).head(10)
    for _, row in top_errors.iterrows():
        plt.text(row['lon'], row['lat'], row['city'], fontsize=8)

    country_centers = plot_df.groupby('country')[['lon', 'lat']].mean().reset_index()
    countries_to_label = ['Brazil', 'Australia', 'India', 'China', 'United States']
    filtered = country_centers[country_centers['country'].isin(countries_to_label)]
    for _, row in filtered.iterrows():
        plt.text(row['lon'], row['lat'], row['country'], fontsize=10, weight='bold')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Residual Map")
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/geographic_residual_map.png", dpi=300)
    plt.show()

    # Opened sphere map with adaptive geodesic conformal uncertainty
    plot_opened_sphere_map(
        theta_grid=theta_grid,
        phi_grid=phi_grid,
        temperature=reconstruction.real,
        uncertainty=uncertainty,
        theta_cities=theta_data,
        phi_cities=phi_data,
        city_values=pred_data,
        city_uncertainty=sigma_city,
        city_names=data["City"].values,
        country_names=data["Country"].values,
        label_uncertainty_percentile=high_uncertainty_percentile,
        max_labels=high_uncertainty_max_labels,
        cut_longitude=0,
        name="opened_globe_uncertainty",
        title="Predicted Adaptive Geodesic Conformal Uncertainty Overlay",
    )
    harmonic_method.plot_prediction_and_uncertainty(
        theta_grid,
        phi_grid,
        reconstruction.real,
        uncertainty,
        theta_data,
        phi_data,
        pred_data,
    )

def main():
    """Script entry point for user interaction."""
    best_grad = None
    while True:
        menu = input('(1) Train model, (2) Reconstruct data, (3) Degree robustness experiment, (4) Exit: ')
        if menu == '1':
            model, best_grad = train_model()
            print(
                f"\nBest gradient available for degree robustness: "
                f"{best_grad}"
            )
        elif menu == '2':
            n = input('Enter the number of points to use to create an n^2 sized grid for field reconstruction: ')
            if int(n) <= 0:
                print('Invalid choice. n cannot be less than or equal to 0.')
                continue
            reconstruct_field(int(n))
        elif menu == '3':
            if best_grad is None:
                print(
                    "\nPlease train the model first "
                    "before running the degree robustness experiment."
                )
                continue
            print(
                f"\nRunning degree robustness using best grad = "
                f"{best_grad}"
            )
            run_degree_robustness_experiment(best_grad)
        elif menu == '4':
            break
    
if __name__ == "__main__":
    main()
