## Code Collection for Outliers Detection

############ libraries needed #################
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import umap


##### histogram/boxplot/violin plot #######

def plot_histogram(df: pd.DataFrame,
                               class_col: str,
                               class_val,
                               features,
                               bins: int = 10):
    """
    Plot overlapping histograms with KDE for only the specified classes
    in class_val, for each feature in features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    class_col : str
        Name of the column containing class labels.
    class_val : str or list of str
        One or more class labels to include (e.g. ['Bream', 'Smelt']).
    features : str or list of str
        Feature name or list of feature names to plot.
    bins : int, optional
        Number of bins for the histogram (default is 10).
    """
    # Normalize inputs
    if isinstance(class_val, str):
        class_val = [class_val]
    if isinstance(features, str):
        features = [features]

    # Filter to only the desired classes
    df_sub = df[df[class_col].isin(class_val)]

    for feat in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(
            data=df_sub,
            x=feat,
            hue=class_col,
            bins=bins,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.4
        )
        plt.title(f'Overlapping distributions of {feat} for {class_val}')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()
def plot_boxplots(df: pd.DataFrame,
                  class_col: str,
                  class_val,
                  features):
    """
    Plot a boxplot for each feature in `features`, restricted to rows
    where df[class_col] == class_val, using Seaborn.
    """
    if isinstance(features, str):
        features = [features]

    sub_df = df[df[class_col] == class_val]
    for feat in features:
        plt.figure(figsize=(4, 6))
        sns.boxplot(y=sub_df[feat])
        plt.title(f'Boxplot of {feat} for {class_val}')
        plt.ylabel(feat)
        plt.tight_layout()
        plt.show()
def plot_distributions(df, features):
    """
    Plots violin + box + strip plots for each feature in `features` from DataFrame `df`.
    """
    sns.set(style="whitegrid")
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Distribution of: {', '.join(features)}", fontsize=20)

    for ax, feat in zip(axes, features):
        sns.violinplot(y=df[feat], ax=ax, inner=None, color="lightblue")
        sns.boxplot(
            y=df[feat], ax=ax, width=0.2,
            boxprops={"facecolor": "white", "edgecolor": "black"},
            medianprops={"color": "red"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"}
        )
        sns.stripplot(y=df[feat], ax=ax, color="green", size=5, jitter=True, alpha=0.5)
        ax.set_xlabel(feat)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_IQR_Zscore(df, method='IQR'):
    """
    Detects outliers in a DataFrame using IQR or Z-score.
    Returns outlier rows, summary of counts per feature, and optionally shows plots.

    Parameters:
        df (pd.DataFrame): A numeric subset of the full dataset (e.g., one class).
        method (str): 'IQR' or 'Z-score'.
        plot (bool): Whether to show annotated plots for each feature.

    Returns:
        outliers_df (pd.DataFrame): Rows identified as outliers.
        outlier_summary (pd.DataFrame): Count of outliers per feature.
    """
    df_subset = df.copy()
    numeric_cols = df_subset.select_dtypes(include=np.number).columns

    outlier_mask = pd.Series(False, index=df_subset.index)
    outlier_counts = {}

    for col in numeric_cols:
        if method.upper() == 'IQR':
            Q1 = df_subset[col].quantile(0.25)
            Q3 = df_subset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df_subset[col] < lower) | (df_subset[col] > upper)
        elif method.upper() == 'Z-SCORE':
            col_z = zscore(df_subset[col], nan_policy='omit')
            outliers = np.abs(col_z) > 3
        else:
            raise ValueError("Method must be either 'IQR' or 'Z-score'")

        outlier_mask |= outliers
        outlier_counts[col] = outliers.sum()      

    # Create summary DataFrame
    outlier_summary = pd.DataFrame({
        'Feature': list(outlier_counts.keys()),
        'Outlier_Count': list(outlier_counts.values())
    })

    return df_subset[outlier_mask], outlier_summary


######## Outliers detectors ###############
def detect_outliers_IQR_Zscore(df, method='IQR'):
    """
    Detects outliers in a DataFrame using IQR or Z-score.
    Returns:
        - a combined DataFrame of outlier rows per feature (with multi-index),
        - a combined DataFrame of all outlier rows (deduplicated),
        - a summary DataFrame of outlier counts per feature.

    Parameters:
        df (pd.DataFrame): A numeric subset of the full dataset (e.g., one class).
        method (str): 'IQR' or 'Z-score'.

    Returns:
        outliers_by_feature_df (pd.DataFrame): Multi-index DataFrame of outliers per feature.
        combined_outliers_df (pd.DataFrame): Unique rows that are outliers in any feature.
        outlier_summary (pd.DataFrame): Count of outliers per feature.
    """
    
    df_subset = df.copy()
    numeric_cols = df_subset.select_dtypes(include=np.number).columns

    outliers_by_feature = {}
    outlier_mask = pd.Series(False, index=df_subset.index)
    outlier_counts = {}

    for col in numeric_cols:
        if method.upper() == 'IQR':
            Q1 = df_subset[col].quantile(0.25)
            Q3 = df_subset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df_subset[col] < lower) | (df_subset[col] > upper)
        elif method.upper() == 'Z-SCORE':
            col_z = zscore(df_subset[col], nan_policy='omit')
            outliers = np.abs(col_z) > 3
        else:
            raise ValueError("Method must be either 'IQR' or 'Z-score'")

        outlier_counts[col] = outliers.sum()
        outliers_by_feature[col] = df_subset[outliers]
        outlier_mask |= outliers

    # Combine all outliers per feature into a multi-index DataFrame
    outliers_by_feature_df = pd.concat(
        outliers_by_feature.values(),
        keys=outliers_by_feature.keys(),
        names=["Feature", "Index"]
    )

    # Get deduplicated combined outliers
    combined_outliers_df = df_subset[outlier_mask].drop_duplicates()

    # Create summary
    outlier_summary = pd.DataFrame({
        'Feature': list(outlier_counts.keys()),
        'Outlier_Count': list(outlier_counts.values())
    })

    return outliers_by_feature_df, combined_outliers_df, outlier_summary
def detect_outliers_isolation_forest(df, X, contamination=0.1, random_state=42):
    """
    Detect outliers in a DataFrame using Isolation Forest.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    X : the features     
    contamination : float, default=0.1
        The proportion of outliers in the data set.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    df_with_labels : pandas.DataFrame
        Original DataFrame with an added 'ISO_Outlier' column
        (1 = inlier, -1 = outlier).
    outliers_df : pandas.DataFrame
        Subset of `df_with_labels` where 'ISO_Outlier' == -1.
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Fit Isolation Forest
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state)
    y_pred = iso.fit_predict(X_scaled)

    # 2. Attach predictions to a copy of the DataFrame
    df_with_labels = df.copy()
    df_with_labels['ISO_Outlier'] = y_pred

    # 3. Extract only the outliers
    outliers_df = df_with_labels[df_with_labels['ISO_Outlier'] == -1]    

    return df_with_labels, outliers_df
def detect_outliers_lof(df, X, n_neighbors=20, contamination=0.1):
    """
    Detect outliers in a DataFrame using Local Outlier Factor.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    
    X : the features  
    
    n_neighbors : int, default=20
        Number of neighbors to use by default for kneighbors queries.
    contamination : float, default=0.1
        The proportion of outliers in the data set.

    Returns
    -------
    df_with_labels : pandas.DataFrame
        Original DataFrame with an added 'LOF_Outlier' column
        (1 = inlier, -1 = outlier).
    outliers_df : pandas.DataFrame
        Subset of `df_with_labels` where 'LOF_Outlier' == -1.
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Fit LOF and predict labels
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    y_pred = lof.fit_predict(X_scaled)

    # 2. Attach predictions to a copy of the DataFrame
    df_with_labels = df.copy()
    df_with_labels['LOF_Outlier'] = y_pred

    # 3. Extract only the outliers
    outliers_df = df_with_labels[df_with_labels['LOF_Outlier'] == -1]

    return df_with_labels, outliers_df
def detect_outliers_ocsvm(df, X, nu=0.1, kernel="rbf", gamma="scale"):
    """
    Detect outliers in a DataFrame using One-Class SVM.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    X : features
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a lower
        bound of the fraction of support vectors.
    kernel : str, default="rbf"
        Kernel type to be used in the algorithm.
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel coefficient.

    Returns
    -------
    df_with_labels : pandas.DataFrame
        Original DataFrame with an added 'OCSVM_Outlier' column
        (1 = inlier, -1 = outlier).
    outliers_df : pandas.DataFrame
        Subset of `df_with_labels` where 'OCSVM_Outlier' == -1.
    """
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Fit One-Class SVM and predict labels
    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    y_pred = oc_svm.fit_predict(X_scaled)

    # 2. Attach predictions to a copy of the DataFrame
    df_with_labels = df.copy()
    df_with_labels['OCSVM_Outlier'] = y_pred

    # 3. Extract only the outliers
    outliers_df = df_with_labels[df_with_labels['OCSVM_Outlier'] == -1]

    return df_with_labels, outliers_df
def detect_outliers_dbscan(df ,X, eps=0.5, min_samples=5, scale=True):
    """
    Detect outliers (noise points) in a DataFrame using DBSCAN.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    
    X : features
    
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    scale : bool, default=True
        Whether to standardize features before DBSCAN.

    Returns
    -------
    df_with_labels : pandas.DataFrame
        Original DataFrame with an added 'DBSCAN_Cluster' column
        (–1 = noise/outlier, 0,1,2,… = cluster labels).
    outliers_df : pandas.DataFrame
        Subset of `df_with_labels` where 'DBSCAN_Cluster' == -1.
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Fit DBSCAN and predict labels
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = db.fit_predict(X_scaled)

    # 4. Attach predictions to a copy of the DataFrame
    df_with_labels = df.copy()
    df_with_labels['DBSCAN_Cluster'] = y_pred

    # 5. Extract only the noise points (outliers)
    outliers_df = df_with_labels[df_with_labels['DBSCAN_Cluster'] == -1]

    return df_with_labels, outliers_df
def detect_outliers_autoencoder(
    df,
    X,
    contamination=0.1,
    seed=42,
    encoding_dim_1=64,
    encoding_dim_2=32,
    epochs=50,
    batch_size=32,
    verbose=0
):
    """
    Detect outliers in a DataFrame using an Autoencoder.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    
    X  : features
    
    contamination : float, default=0.1
        The proportion of outliers in the data set.
    seed : int, default=42
        Seed for numpy and TensorFlow reproducibility.
    encoding_dim_1 : int, default=64
        Number of neurons in the first encoding layer.
    encoding_dim_2 : int, default=32
        Number of neurons in the second (bottleneck) layer.
    epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Training batch size.
    verbose : int, default=0
        Verbosity level for model.fit().

    Returns
    -------
    df_with_labels : pandas.DataFrame
        Original DataFrame with an added 'Autoencoder_Outlier' column
        (1 = inlier, -1 = outlier).
    outliers_df : pandas.DataFrame
        Subset of `df_with_labels` where 'Autoencoder_Outlier' == -1.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    # 1. Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)


    # 2. Build the autoencoder
    input_dim = X_scaled.shape[1]
    model = Sequential([
        Dense(encoding_dim_1, activation='relu', input_dim=input_dim),
        Dense(encoding_dim_2, activation='relu'),
        Dense(encoding_dim_1, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 4. Train
    model.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_scaled, X_scaled),
        shuffle=False,
        verbose=verbose
    )

    # 5. Compute reconstruction errors
    reconstructions = model.predict(X_scaled)
    reconstruction_errors = np.mean(np.abs(X_scaled - reconstructions), axis=1)

    # 6. Determine threshold & label
    threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
    labels = np.where(reconstruction_errors > threshold, -1, 1)

    # 7. Attach to DataFrame
    df_with_labels = df.copy()
    df_with_labels['Autoencoder_Outlier'] = labels
    outliers_df = df_with_labels[df_with_labels['Autoencoder_Outlier'] == -1]

    return df_with_labels, outliers_df
def detect_outliers_all_method(df,X,
    random_state=42,
    contamination=0.1,
    lof_n_neighbors=20,
    ocsvm_nu=0.1,
    ocsvm_kernel="rbf",
    ocsvm_gamma="scale",
    dbscan_eps=0.5,
    dbscan_min_samples=5,
    ae_encoding_dim_1=64,
    ae_encoding_dim_2=32,
    ae_epochs=50,
    ae_batch_size=32,
    ae_seed=42,
    ae_verbose=0
):

    # 1. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_result = df.copy()

    # 2. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df_result['ISO_Outlier'] = iso.fit_predict(X_scaled)

    # 3. LOF
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=contamination)
    df_result['LOF_Outlier'] = lof.fit_predict(X_scaled)

    # 4. One-Class SVM
    ocsvm = OneClassSVM(nu=ocsvm_nu, kernel=ocsvm_kernel, gamma=ocsvm_gamma)
    df_result['OCSVM_Outlier'] = ocsvm.fit_predict(X_scaled)

    # 5. DBSCAN
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    df_result['DBSCAN_Cluster'] = db.fit_predict(X_scaled)
#     df_result['DBSCAN_Outlier'] = np.where(df_result['DBSCAN_Cluster'] == -1, -1, 1)

    # 6. Autoencoder
    # eliminate randomness (mostly, not 100 percent)
    np.random.seed(ae_seed)
    tf.random.set_seed(ae_seed)

    input_dim = X_scaled.shape[1]
    ae = Sequential([
        Dense(ae_encoding_dim_1, activation='relu', input_dim=input_dim),
        Dense(ae_encoding_dim_2, activation='relu'),
        Dense(ae_encoding_dim_1, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(X_scaled, X_scaled,
           epochs=ae_epochs,
           batch_size=ae_batch_size,
           shuffle=True,
           verbose=ae_verbose)

    reconstructions = ae.predict(X_scaled)
    errors = np.mean(np.abs(X_scaled - reconstructions), axis=1)
    threshold = np.percentile(errors, 100 * (1 - contamination))
    df_result['Autoencoder_Outlier'] = np.where(errors > threshold, -1, 1)

    # 7. Consensus Voting (majority or at least 2 methods vote)
    outlier_columns = [
        'ISO_Outlier',
        'LOF_Outlier',
        'OCSVM_Outlier',
        'DBSCAN_Cluster',
        'Autoencoder_Outlier'
    ]
    df_result['Outlier_Votes'] = df_result[outlier_columns].apply(lambda row: list(row).count(-1), axis=1)
    df_result['Consensus_Outlier'] = df_result['Outlier_Votes'].apply(lambda x: -1 if x >= 2 else 1)

    # 8. Filter only outliers
    outliers_df = df_result[df_result['Consensus_Outlier'] == -1]

    return df_result, outliers_df

######## Dimension Reduction Techniques ##############
def detect_outliers_pca(X, n_components=2, method='percentile', threshold=95, visualize=True): 

    """
    Detects outliers using PCA reconstruction error with percentile, Z-score, or IQR thresholding.

    Parameters:
        X (pd.DataFrame): Input DataFrame with numeric features.
        n_components (int): Number of PCA components (2 or 3 for visualization).
        method (str): 'percentile', 'zscore', or 'iqr' for outlier detection strategy.
        
    
        threshold (float): Threshold for outlier detection:
                           - Percentile (e.g. 95)
                           - Z-score cutoff (e.g. 3)
                           - Not used for IQR (uses 1.5 * IQR internally)
                           
                           
        visualize (bool): Whether to show a 2D/3D PCA plot.

    Returns:
        outliers_df (pd.DataFrame): Outlier rows from X.
        visualization (np.ndarray): PCA-transformed values.
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_projected = pca.inverse_transform(X_pca)

    # Reconstruction error
    reconstruction_error = np.mean((X_scaled - X_projected) ** 2, axis=1)

    # Outlier detection method
    if method == 'percentile':
        cutoff = np.percentile(reconstruction_error, threshold)
        outliers = reconstruction_error > cutoff
    elif method == 'zscore':
        z_scores = zscore(reconstruction_error)
        outliers = np.abs(z_scores) > threshold
    elif method == 'iqr':
        Q1 = np.percentile(reconstruction_error, 25)
        Q3 = np.percentile(reconstruction_error, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = reconstruction_error > upper_bound
    else:
        raise ValueError("method must be 'percentile', 'zscore', or 'iqr'")

    # Visualization (optional)
    if visualize:
        fig = plt.figure()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[~outliers, 0], X_pca[~outliers, 1], X_pca[~outliers, 2], c='blue', label='Inliers')
            ax.scatter(X_pca[outliers, 0], X_pca[outliers, 1], X_pca[outliers, 2], c='red', label='Outliers')
            ax.set_title("3D PCA-based Outlier Detection")
        elif n_components == 2:
            plt.scatter(X_pca[~outliers, 0], X_pca[~outliers, 1], c='blue', label='Inliers')
            plt.scatter(X_pca[outliers, 0], X_pca[outliers, 1], c='red', label='Outliers')
            plt.title("2D PCA-based Outlier Detection")
        else:
            raise ValueError("Visualization only supported for 2 or 3 PCA components.")
        plt.legend()
        plt.show()

    outliers_df = X[outliers]

    return outliers_df
def detect_outliers_pca_all_method(df, X, methods=["isolation_forest"], ndim=2, visualize=True, **kwargs):
    result_df = df.copy()
    X_scaled = StandardScaler().fit_transform(X)

    # PCA transformation
    pca = PCA(n_components=ndim)
    X_pca = pca.fit_transform(X_scaled)

    outlier_flags = []

    for method in methods:
        label_col = f"outlier_{method}"
        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=kwargs.get("contamination", 0.1),
                random_state=42
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "ocsvm":
            clf = OneClassSVM(
                kernel=kwargs.get("kernel", "rbf"),
                nu=kwargs.get("nu", 0.05),
                gamma=kwargs.get("gamma", "scale")
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "dbscan":
            clf = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5)
            )
            labels = clf.fit_predict(X_scaled)
            result_df[label_col] = labels == -1

        elif method == "lof":
            clf = LocalOutlierFactor(
                n_neighbors=kwargs.get("n_neighbors", 20),
                contamination=kwargs.get("contamination", 0.05)
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "autoencoder":
            input_dim = X_scaled.shape[1]
            encoding_dim = kwargs.get("encoding_dim", min(10, input_dim // 2))

            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='linear')(encoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            autoencoder.fit(X_scaled, X_scaled, epochs=kwargs.get("epochs", 50),
                            batch_size=kwargs.get("batch_size", 32), verbose=0)
            reconstructions = autoencoder.predict(X_scaled, verbose=0)
            loss = np.mean((X_scaled - reconstructions) ** 2, axis=1)
            threshold = np.percentile(loss, kwargs.get("threshold_percentile", 95))
            result_df[label_col] = loss > threshold

        outlier_flags.append(result_df[label_col])

    # Combine all outlier flags (use any method's outlier tag as criteria)
    combined_outliers = np.logical_or.reduce(outlier_flags)
    outliers_df = result_df[combined_outliers].copy()

    # Visualization
    if visualize:
        if ndim == 2:
            plt.figure(figsize=(10, 6))
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                plt.scatter(X_pca[~mask, 0], X_pca[~mask, 1], alpha=0.6, label=f'{method} - Inlier')
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.8, marker='x', label=f'{method} - Outlier')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA + Outlier Detection")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif ndim == 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                ax.scatter(X_pca[~mask, 0], X_pca[~mask, 1], X_pca[~mask, 2], alpha=0.6, label=f'{method} - Inlier')
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], alpha=0.9, marker='x', label=f'{method} - Outlier')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("3D PCA + Outlier Detection")
            ax.legend()
            plt.show()

    return outliers_df
def tsne_visualize(df, x, n_components=2, perplexity=30, random_state=42):
    """
    Standardizes the features and visualizes data using t-SNE in 2D or 3D.

    Parameters:
    - df: DataFrame (can be used for labels or future enhancements)
    - x: Raw input features (will be standardized inside the function)
    - n_components: 2 or 3 (for 2D or 3D visualization)
    - perplexity: t-SNE perplexity (default 30)
    - random_state: Random seed for reproducibility (default 42)
    """
    # Step 1: Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Step 2: Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init='pca'
    )
    x_tsne = tsne.fit_transform(x_scaled)

    # Step 3: Plot
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=5)
        plt.title("t-SNE 2D Visualization")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], s=5)
        ax.set_title("t-SNE 3D Visualization")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.show()
    else:
        raise ValueError("n_components must be either 2 or 3")
def detect_outliers_tsne_all_method(df, X, 
                                       methods=["isolation_forest"],
                                       ndim=2, 
                                       visualize=True,
                                       perplexity=30, 
                                       random_state=42, **kwargs):

    result_df = df.copy()
    X_scaled = StandardScaler().fit_transform(X)

    # t-SNE transformation
    tsne = TSNE(n_components=ndim, perplexity=perplexity, random_state=random_state, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)

    outlier_flags = []

    for method in methods:
        label_col = f"outlier_{method}"
        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=kwargs.get("contamination", 0.1),
                random_state=random_state
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "ocsvm":
            clf = OneClassSVM(
                kernel=kwargs.get("kernel", "rbf"),
                nu=kwargs.get("nu", 0.05),
                gamma=kwargs.get("gamma", "scale")
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "dbscan":
            clf = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5)
            )
            labels = clf.fit_predict(X_scaled)
            result_df[label_col] = labels == -1

        elif method == "lof":
            clf = LocalOutlierFactor(
                n_neighbors=kwargs.get("n_neighbors", 20),
                contamination=kwargs.get("contamination", 0.05)
            )
            result_df[label_col] = clf.fit_predict(X_scaled) == -1

        elif method == "autoencoder":
            input_dim = X_scaled.shape[1]
            encoding_dim = kwargs.get("encoding_dim", min(10, input_dim // 2))

            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='linear')(encoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            autoencoder.fit(X_scaled, X_scaled, epochs=kwargs.get("epochs", 50),
                            batch_size=kwargs.get("batch_size", 32), verbose=0)
            reconstructions = autoencoder.predict(X_scaled, verbose=0)
            loss = np.mean((X_scaled - reconstructions) ** 2, axis=1)
            threshold = np.percentile(loss, kwargs.get("threshold_percentile", 95))
            result_df[label_col] = loss > threshold

        outlier_flags.append(result_df[label_col])

    # Combine all outlier flags (logical OR)
    combined_outliers = np.logical_or.reduce(outlier_flags)
    outliers_df = result_df[combined_outliers].copy()

    # Visualization
    if visualize:
        if ndim == 2:
            plt.figure(figsize=(10, 6))
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                plt.scatter(X_tsne[~mask, 0], X_tsne[~mask, 1], alpha=0.6, label=f'{method} - Inlier')
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.8, marker='x', label=f'{method} - Outlier')
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title("t-SNE + Outlier Detection")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif ndim == 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                ax.scatter(X_tsne[~mask, 0], X_tsne[~mask, 1], X_tsne[~mask, 2], alpha=0.6, label=f'{method} - Inlier')
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2], alpha=0.9, marker='x', label=f'{method} - Outlier')
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            ax.set_title("3D t-SNE + Outlier Detection")
            ax.legend()
            plt.show()

    return outliers_df
def detect_outliers_umap_all_method(df, X,
                           ndim=2,
                           visualize=True,
                           n_neighbors=15,
                           min_dist=0.1,
                           metric="euclidean",
                           methods=["isolation_forest"],
                           random_state=42, # eliminate randomness im  umap
                           **kwargs):
    """
    UMAP-based outlier detection with multiple methods and 2D/3D visualization.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        X (array-like): Input features.
        ndim (int): UMAP output dimension (2 or 3).
        visualize (bool): Whether to visualize in 2D/3D.
        n_neighbors, min_dist, metric: UMAP parameters.
        methods (list): Outlier detection methods.
        random_state (int): Reproducibility.
        **kwargs: Additional params for detection models.

    Returns:
        pd.DataFrame: Subset of original DataFrame with rows flagged as outliers.
    """
    result_df = df.copy()
    X_scaled = StandardScaler().fit_transform(X)

    # UMAP projection
    reducer = umap.UMAP(
        n_components=ndim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X_scaled)

    outlier_flags = []

    for method in methods:
        col = f"outlier_{method}"
        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=kwargs.get("contamination", 0.1),
                random_state=random_state
            )
            result_df[col] = clf.fit_predict(X_scaled) == -1

        elif method == "ocsvm":
            clf = OneClassSVM(
                kernel=kwargs.get("kernel", "rbf"),
                nu=kwargs.get("nu", 0.05),
                gamma=kwargs.get("gamma", "scale")
            )
            result_df[col] = clf.fit_predict(X_scaled) == -1

        elif method == "dbscan":
            clf = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5)
            )
            labels = clf.fit_predict(X_scaled)
            result_df[col] = labels == -1
            
        elif method == "lof":
            clf = LocalOutlierFactor(
                n_neighbors=kwargs.get("lof_n_neighbors", 20),
                contamination=kwargs.get("contamination", 0.05)
            )
            result_df[col] = clf.fit_predict(X_scaled) == -1

        elif method == "autoencoder":
            input_dim = X_scaled.shape[1]
            encoding_dim = kwargs.get("encoding_dim", min(10, input_dim // 2))

            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='linear')(encoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            autoencoder.fit(X_scaled, X_scaled, epochs=kwargs.get("epochs", 50),
                            batch_size=kwargs.get("batch_size", 32), verbose=0)
            reconstructions = autoencoder.predict(X_scaled, verbose=0)
            loss = np.mean((X_scaled - reconstructions) ** 2, axis=1)
            threshold = np.percentile(loss, kwargs.get("threshold_percentile", 95))
            result_df[col] = loss > threshold

        outlier_flags.append(result_df[col])

    # Combine outlier flags (logical OR across all methods)
    combined_outliers = np.logical_or.reduce(outlier_flags)
    outliers_df = result_df.loc[combined_outliers, df.columns.tolist() + [f"outlier_{m}" for m in methods]]

    # Visualization
    if visualize:
        if ndim == 2:
            plt.figure(figsize=(10, 6))
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                plt.scatter(X_umap[~mask, 0], X_umap[~mask, 1], alpha=0.6, label=f'{method} - Inlier')
                plt.scatter(X_umap[mask, 0], X_umap[mask, 1], alpha=0.8, marker='x', label=f'{method} - Outlier')
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("2D UMAP + Outlier Detection")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif ndim == 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for method in methods:
                mask = result_df[f"outlier_{method}"]
                ax.scatter(X_umap[~mask, 0], X_umap[~mask, 1], X_umap[~mask, 2],
                           alpha=0.6, label=f'{method} - Inlier')
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2],
                           alpha=0.9, marker='x', label=f'{method} - Outlier')
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_zlabel("UMAP 3")
            ax.set_title("3D UMAP + Outlier Detection")
            ax.legend()
            plt.show()

    return outliers_df

#################### fin ######## by: kaikuh pogi ################

