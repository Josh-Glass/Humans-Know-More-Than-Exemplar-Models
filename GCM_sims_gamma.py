import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress standard math/plotting warnings for a clean console output
warnings.filterwarnings('ignore')

# ==========================================
# 0. SETUP DIRECTORY
# ==========================================
out_dir = "gcm_sim_img"
os.makedirs(out_dir, exist_ok=True)

# ==========================================
# 1. DEFINE CATEGORY COORDINATES
# ==========================================
full_circle_A = np.array([[.3,.1], [.3,.9], [.7,.1], [.7,.9], [.9,.3], [.9,.5], [.1,.5], [.1,.7]])
full_circle_B = np.array([[.3,.3], [.5,.3], [.7,.3], [.3,.5], [.3,.7], [.7,.5], [.7,.7], [.5,.7]])

curtailed_circle_A = np.array([[.7,.1], [.7,.9], [.9,.3], [.9,.5]])
curtailed_circle_B = np.array([[.5,.3], [.7,.3], [.7,.5], [.7,.7]])

uniVproxi_A = np.array([[.9,1], [0,.8], [.9,.6]])
uniVproxi_B = np.array([[0,.4], [0,.2], [0,0]])

learn_extra_A = np.array([[.2,0],[.2,.1],[.2,.2],[.2,.3],[.2,.4],[.2,.5],[.45,0],[.45,.1],[.45,.2],[.45,.3],[.45,.4],[.45,.5]])
learn_extra_B = np.array([[.55,.4],[.55,.5],[.55,.6],[.55,.7],[.55,.8],[.55,.9],[.8,.4],[.8,.5],[.8,.6],[.8,.7],[.8,.8],[.8,.9]])

tmb_A = np.array([[0.4, 0.7], [0.5, 0.7], [0.9, 0.7], [0.5, 0.45], [0.7, 0.55], [0.9, 0.45]])
tmb_B = np.array([[0.1, 0.55], [0.2, 0.45], [0.6, 0.55], [0.35, 0.3], [0.5, 0.3], [0.6, 0.3]])

conditions_list = [
    {"name": "Full Circle", "A": full_circle_A, "B": full_circle_B, "test": np.array([0.9, 0.7]), "desc": "T0 (0.9, 0.7)"},
    {"name": "Curtailed Circle", "A": curtailed_circle_A, "B": curtailed_circle_B, "test": np.array([0.9, 0.7]), "desc": "T0 (0.9, 0.7)"},
    {"name": "Uni vs Proxi", "A": uniVproxi_A, "B": uniVproxi_B, "test": np.array([0.9, 0.4]), "desc": "T1 (0.9, 0.4)"},
    {"name": "Learn Extra", "A": learn_extra_A, "B": learn_extra_B, "test": np.array([0.6, 0.0]), "desc": "T1 (0.6, 0.0)"},
    {"name": "Top Middle Bottom", "A": tmb_A, "B": tmb_B, "test": np.array([0.1, 0.7]), "desc": "T0 (0.1, 0.7)"}
]

# ==========================================
# 2. DEFINE THE GCM FUNCTIONS
# ==========================================
def gcm_predict_alpha(test_pt, catA, catB, c, w, gamma):
    dist_A = w * np.abs(catA[:, 0] - test_pt[0]) + (1 - w) * np.abs(catA[:, 1] - test_pt[1])
    dist_B = w * np.abs(catB[:, 0] - test_pt[0]) + (1 - w) * np.abs(catB[:, 1] - test_pt[1])
    
    sim_A = np.sum(np.exp(-c * dist_A))
    sim_B = np.sum(np.exp(-c * dist_B))
    
    if sim_A == 0 and sim_B == 0: return 0.5
    
    scaled_sim_A = sim_A ** gamma
    scaled_sim_B = sim_B ** gamma
    
    if scaled_sim_A == 0 and scaled_sim_B == 0: return 0.5
    
    return scaled_sim_A / (scaled_sim_A + scaled_sim_B)

def gcm_training_accuracy(catA, catB, c, w, gamma):
    accs = []
    for a in catA: accs.append(gcm_predict_alpha(a, catA, catB, c, w, gamma))
    for b in catB: accs.append(1 - gcm_predict_alpha(b, catA, catB, c, w, gamma))
    return np.mean(accs)

# ==========================================
# 3. MAIN GRID SEARCH (Histograms & Base Heatmaps)
# ==========================================
def run_gcm_grid_search(apply_filter=True, search_gamma=False):
    w_vals = np.linspace(0.0, 1.0, 40)
    c_vals = np.linspace(0.1, 15.0, 40)
    
    if search_gamma:
        gamma_vals = np.linspace(0.5, 5.0, 15) 
        gamma_status = "0.5 to 5.0 (15 steps)"
        title_suffix = "with Gamma Grid Search"
        file_suffix = "_3D_gamma_search"
    else:
        gamma_vals = [1.0] 
        gamma_status = "1.0 (Fixed, No Scaling)"
        title_suffix = "(Standard, No Gamma)"
        file_suffix = "_2D_no_gamma"

    filter_status = ">80% Accuracy Required" if apply_filter else "None (Unfiltered)"
    
    info_text = (
        "Simulation Parameters:\n\n"
        "• Sensitivity (c): 0.1 to 15.0 (40 steps)\n"
        "• Attention (w): 0.0 to 1.0 (40 steps)\n"
        f"• Scaling (gamma): {gamma_status}\n"
        "• Spatial Metric: City Block (L1)\n"
        f"• Accuracy Filter: {filter_status}"
    )

    print(f"Running primary grid search: {title_suffix}...")
    
    hist_data = []
    heatmap_Z_grids = [] 

    for cond in conditions_list:
        Z = np.zeros((len(c_vals), len(w_vals)))
        valid_count = 0
        
        for i, c in enumerate(c_vals):
            for j, w in enumerate(w_vals):
                for g in gamma_vals:
                    if apply_filter:
                        train_acc = gcm_training_accuracy(cond["A"], cond["B"], c, w, g)
                        is_valid = train_acc >= 0.80
                    else:
                        is_valid = True
                    
                    if is_valid:
                        p_alpha = gcm_predict_alpha(cond["test"], cond["A"], cond["B"], c, w, g)
                        hist_data.append({"Condition": cond["name"], "Desc": cond["desc"], "P_Alpha": p_alpha})
                        if not search_gamma:
                            Z[i, j] = p_alpha
                            valid_count += 1
                    else:
                        if not search_gamma:
                            Z[i, j] = np.nan
                            
        if not search_gamma:
            heatmap_Z_grids.append((Z, valid_count))

    filter_title = "(>80% Acc)" if apply_filter else "(Unfiltered)"

    # --- HEATMAPS (Only for 2D Search) ---
    if not search_gamma:
        fig_h, axes_h = plt.subplots(2, 3, figsize=(18, 11))
        axes_h = axes_h.flatten()
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color='lightgrey') 
        W, C = np.meshgrid(w_vals, c_vals)

        for idx, cond in enumerate(conditions_list):
            ax = axes_h[idx]
            Z, v_count = heatmap_Z_grids[idx]
            
            if v_count > 0:
                contour = ax.contourf(W, C, Z, levels=np.linspace(0, 1, 21), cmap=cmap)
                if np.nanmin(Z) <= 0.5 <= np.nanmax(Z):
                    ax.contour(W, C, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)
            else:
                ax.set_facecolor('lightgrey')
                ax.text(0.5, 0.5, 'ZERO Valid Models', ha='center', va='center', fontsize=14, color='red', transform=ax.transAxes)

            ax.set_title(f'{cond["name"]}\n{cond["desc"]}', fontweight='bold')
            ax.set_xlabel('Attention to Dim 1 (w)')
            ax.set_ylabel('Sensitivity (c)')

        info_ax_h = axes_h[-1]
        info_ax_h.axis('off')
        info_ax_h.text(0.05, 0.95, info_text, fontsize=13, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=1.2", fc="whitesmoke", ec="gray", lw=1.5))

        if 'contour' in locals():
            cbar_ax = info_ax_h.inset_axes([0.05, 0.15, 0.9, 0.08])
            fig_h.colorbar(contour, cax=cbar_ax, orientation='horizontal').set_label('Predicted P(Alpha)', fontsize=12)

        fig_h.suptitle(f'GCM Parameter Space Predictions {filter_title} {title_suffix}', fontsize=20, fontweight='bold')
        fig_h.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        plt.savefig(os.path.join(out_dir, f'gcm_heatmaps_{"filtered" if apply_filter else "unfiltered"}{file_suffix}.png'), dpi=300)
        plt.close(fig_h) 

    # --- HISTOGRAMS (For both 2D and 3D Searches) ---
    df_hist = pd.DataFrame(hist_data)
    fig_d, axes_d = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    axes_d = axes_d.flatten()

    for idx, cond in enumerate(conditions_list):
        ax = axes_d[idx]
        if not df_hist.empty:
            data = df_hist[df_hist["Condition"] == cond["name"]]
            if not data.empty:
                sns.histplot(data=data, x="P_Alpha", bins=20, kde=True, ax=ax, 
                             color="steelblue", edgecolor="black", alpha=0.6)
        
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2.5)
        ax.set_title(f'{cond["name"]}\n{cond["desc"]}', fontweight='bold')
        ax.set_xlabel('Predicted P(Alpha)')
        ax.set_xlim(-0.05, 1.05)

    info_ax_d = axes_d[-1]
    info_ax_d.axis('off')
    info_ax_d.text(0.05, 0.5, info_text, fontsize=13, va='center', ha='left',
                   bbox=dict(boxstyle="round,pad=1.2", fc="whitesmoke", ec="gray", lw=1.5))

    fig_d.suptitle(f'Density of GCM Predictions {filter_title}\n{title_suffix}', fontsize=20, fontweight='bold')
    fig_d.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    plt.savefig(os.path.join(out_dir, f'gcm_histograms_{"filtered" if apply_filter else "unfiltered"}{file_suffix}.png'), dpi=300)
    plt.close(fig_d)

# ==========================================
# 4. DIAGNOSTIC HEATMAPS (Explicit Counting)
# ==========================================
def run_diagnostic_heatmaps(gamma_val):
    print(f"Running diagnostic heatmap for Gamma = {gamma_val}...")
    w_vals = np.linspace(0.0, 1.0, 40)
    c_vals = np.linspace(0.1, 15.0, 40)
    W, C = np.meshgrid(w_vals, c_vals)
    total_models = len(w_vals) * len(c_vals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='lightgrey') 

    for idx, cond in enumerate(conditions_list):
        ax = axes[idx]
        Z = np.zeros_like(W)
        valid_count = 0
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                train_acc = gcm_training_accuracy(cond["A"], cond["B"], C[i, j], W[i, j], gamma_val)
                if train_acc >= 0.80:
                    Z[i, j] = gcm_predict_alpha(cond["test"], cond["A"], cond["B"], C[i, j], W[i, j], gamma_val)
                    valid_count += 1
                else:
                    Z[i, j] = np.nan 
                    
        if valid_count > 0:
            contour = ax.contourf(W, C, Z, levels=np.linspace(0, 1, 21), cmap=cmap)
            if np.nanmin(Z) <= 0.5 <= np.nanmax(Z):
                ax.contour(W, C, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)
        else:
            ax.set_facecolor('lightgrey')
            ax.text(0.5, 0.5, 'ZERO Valid Models', ha='center', va='center', fontsize=14, color='red', transform=ax.transAxes)

        ax.set_title(f'{cond["name"]}\nValid Models: {valid_count} / {total_models}', fontweight='bold')
        ax.set_xlabel('Attention to Dim 1 (w)')
        ax.set_ylabel('Sensitivity (c)')

    info_ax = axes[-1]
    info_ax.axis('off')
    info_text = f"Diagnostic Heatmaps\nGamma = {gamma_val}\nFilter = >80% Acc"
    info_ax.text(0.05, 0.8, info_text, fontsize=15, va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=1.2", fc="whitesmoke", ec="gray", lw=1.5))
    
    if 'contour' in locals():
        cbar_ax = info_ax.inset_axes([0.05, 0.4, 0.9, 0.08])
        fig.colorbar(contour, cax=cbar_ax, orientation='horizontal').set_label('Predicted P(Alpha)', fontsize=12)

    fig.suptitle(f'Diagnostic Map: Model Viability & Predictions (Gamma = {gamma_val})', fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(os.path.join(out_dir, f'diagnostic_heatmap_gamma_{gamma_val}.png'), dpi=300)
    plt.close(fig)

# ==========================================
# 5. EXECUTE ALL SIMULATIONS
# ==========================================
if __name__ == "__main__":
    print("Starting GCM Simulations...")
    
    # 1. Base 2D Search (No Gamma) -> Heatmaps & Histograms
    run_gcm_grid_search(apply_filter=True, search_gamma=False)
    
    # 2. 3D Gamma Search -> Only Histograms
    run_gcm_grid_search(apply_filter=True, search_gamma=True)
    
    # 3. Diagnostic explicit counts -> Heatmaps
    run_diagnostic_heatmaps(gamma_val=1.0) # Baseline counts
    run_diagnostic_heatmaps(gamma_val=4.0) # Checking for extreme/implausible 'w' usage
    
    print(f"All simulations complete! Check the '{out_dir}' folder for images.")