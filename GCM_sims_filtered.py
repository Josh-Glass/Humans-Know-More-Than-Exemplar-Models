import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# ==========================================
# 2. DEFINE THE GCM FUNCTIONS
# ==========================================
def gcm_predict_alpha(test_pt, catA, catB, c, w):
    dist_A = w * np.abs(catA[:, 0] - test_pt[0]) + (1 - w) * np.abs(catA[:, 1] - test_pt[1])
    dist_B = w * np.abs(catB[:, 0] - test_pt[0]) + (1 - w) * np.abs(catB[:, 1] - test_pt[1])
    
    sim_A = np.sum(np.exp(-c * dist_A))
    sim_B = np.sum(np.exp(-c * dist_B))
    
    if sim_A == 0 and sim_B == 0: return 0.5
    return sim_A / (sim_A + sim_B)

def gcm_training_accuracy(catA, catB, c, w):
    accs = []
    for a in catA: accs.append(gcm_predict_alpha(a, catA, catB, c, w))
    for b in catB: accs.append(1 - gcm_predict_alpha(b, catA, catB, c, w))
    return np.mean(accs)

# ==========================================
# 3. MAIN PLOTTING FUNCTION
# ==========================================
def run_gcm_simulations(apply_filter=True):
    w_vals = np.linspace(0.0, 1.0, 50)
    c_vals = np.linspace(0.1, 15.0, 50)
    W, C = np.meshgrid(w_vals, c_vals)

    conditions = [
        {"name": "Full Circle", "A": full_circle_A, "B": full_circle_B, "test": np.array([0.9, 0.7]), "desc": "T0 (0.9, 0.7)"},
        {"name": "Curtailed Circle", "A": curtailed_circle_A, "B": curtailed_circle_B, "test": np.array([0.9, 0.7]), "desc": "T0 (0.9, 0.7)"},
        {"name": "Uni vs Proxi", "A": uniVproxi_A, "B": uniVproxi_B, "test": np.array([0.9, 0.4]), "desc": "T1 (0.9, 0.4)"},
        {"name": "Learn Extra", "A": learn_extra_A, "B": learn_extra_B, "test": np.array([0.6, 0.0]), "desc": "T1 (0.6, 0.0)"},
        {"name": "Top Middle Bottom", "A": tmb_A, "B": tmb_B, "test": np.array([0.1, 0.7]), "desc": "T0 (0.1, 0.7)"}
    ]

    filter_status = ">80% Accuracy Required" if apply_filter else "None (Unfiltered)"
    info_text = (
        "Simulation Parameters:\n\n"
        "• Sensitivity (c): 0.1 to 15.0 (50 steps)\n"
        "• Attention (w): 0.0 to 1.0 (50 steps)\n"
        "• Total Grid Points: 2,500 per condition\n"
        "• Spatial Metric: City Block (L1)\n"
        f"• Training Filter: {filter_status}"
    )

    # ------------------ HEATMAPS ------------------
    fig_h, axes_h = plt.subplots(2, 3, figsize=(18, 11))
    axes_h = axes_h.flatten()
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='lightgrey') 

    hist_data = []

    for idx, cond in enumerate(conditions):
        ax = axes_h[idx]
        Z = np.zeros_like(W)
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if apply_filter:
                    train_acc = gcm_training_accuracy(cond["A"], cond["B"], C[i, j], W[i, j])
                    is_valid = train_acc >= 0.80
                else:
                    is_valid = True
                
                if is_valid:
                    p_alpha = gcm_predict_alpha(cond["test"], cond["A"], cond["B"], C[i, j], W[i, j])
                    Z[i, j] = p_alpha
                    hist_data.append({"Condition": cond["name"], "Desc": cond["desc"], "P_Alpha": p_alpha})
                else:
                    Z[i, j] = np.nan 
                    
        contour = ax.contourf(W, C, Z, levels=np.linspace(0, 1, 21), cmap=cmap)
        
        if np.nanmin(Z) <= 0.5 <= np.nanmax(Z):
            ax.contour(W, C, Z, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)
            
        ax.set_title(f'{cond["name"]}\n{cond["desc"]}', fontweight='bold')
        ax.set_xlabel('Attention to Dim 1 (w)')
        ax.set_ylabel('Sensitivity (c)')

    # Style the 6th subplot (Info Box + Colorbar)
    info_ax_h = axes_h[-1]
    info_ax_h.axis('off')
    
    # Place text box at the top of the 6th cell
    info_ax_h.text(0.05, 0.95, info_text, fontsize=13, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=1.2", fc="whitesmoke", ec="gray", lw=1.5))

    # Place horizontal colorbar at the bottom of the 6th cell
    cbar_ax = info_ax_h.inset_axes([0.05, 0.15, 0.9, 0.08])
    cbar = fig_h.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Predicted P(Alpha)\n(Grey = Failed Training Criteria)' if apply_filter else 'Predicted P(Alpha)', fontsize=12)

    filter_title = "(>80% Training Accuracy Filter Applied)" if apply_filter else "(Unfiltered)"
    fig_h.suptitle(f'GCM Parameter Space Predictions {filter_title}', fontsize=20, fontweight='bold')
    
    fig_h.tight_layout(rect=[0, 0.03, 1, 0.94])
    file_name_h = f'gcm_heatmaps_{"filtered" if apply_filter else "unfiltered"}.png'
    plt.savefig(os.path.join(out_dir, file_name_h), dpi=300)
    plt.close(fig_h) 

    # ------------------ HISTOGRAMS ------------------
    df_hist = pd.DataFrame(hist_data)
    fig_d, axes_d = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    axes_d = axes_d.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_d[idx]
        data = df_hist[df_hist["Condition"] == cond["name"]]
        
        if not data.empty:
            sns.histplot(data=data, x="P_Alpha", bins=20, kde=True, ax=ax, 
                         color="steelblue", edgecolor="black", alpha=0.6)
        
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2.5)
        ax.set_title(f'{cond["name"]}\n{cond["desc"]}', fontweight='bold')
        ax.set_xlabel('Predicted P(Alpha)')
        ax.set_xlim(-0.05, 1.05)

    # Style the 6th subplot (Info Box centered, no colorbar needed)
    info_ax_d = axes_d[-1]
    info_ax_d.axis('off')
    info_ax_d.text(0.05, 0.5, info_text, fontsize=13, va='center', ha='left',
                   bbox=dict(boxstyle="round,pad=1.2", fc="whitesmoke", ec="gray", lw=1.5))

    fig_d.suptitle(f'Density of GCM Predictions {filter_title}', fontsize=20, fontweight='bold')
    
    fig_d.tight_layout(rect=[0, 0.03, 1, 0.94])
    file_name_d = f'gcm_histograms_{"filtered" if apply_filter else "unfiltered"}.png'
    plt.savefig(os.path.join(out_dir, file_name_d), dpi=300)
    plt.show()

# Run both!
run_gcm_simulations(apply_filter=False)
run_gcm_simulations(apply_filter=True)