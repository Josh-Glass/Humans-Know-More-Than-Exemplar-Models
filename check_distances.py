import numpy as np




#full circle structure coordinates
full_circle_dataA = np.array([
    [.3,.1],
    [.3,.9],
    [.7,.1],
    [.7,.9],

    [.9,.3],
    [.9,.5],

    [.1,.5],
    [.1,.7],
    
]) 
full_circle_dataB = np.array([
    [.3,.3], #0
    [.5,.3], #1
    [.7,.3], #2

    [.3,.5], #3
    [.3,.7], #4

    [.7,.5], #5
    [.7,.7], #6
    [.5,.7], #7

]) 
full_circle_dataT = np.array([
    [.9,.7],
    [0.5,0.1],
    [0.5,0.5],
    [0.5,0.9],
    [.1,.3],
]) 


###############################################
#Curtailed cirlce sturcture coordinates
###############################################
curtailed_circle_dataA = np.array([
    [.7,.1],
    [.7,.9],

    [.9,.3],
    [.9,.5],
]) 
curtailed_circle_dataB = np.array([
    [.5,.3], #1
    [.7,.3], #2
    [.7,.5], #5
    [.7,.7], #6

]) 
curtailed_circle_dataT = np.array([
    [.9,.7],
    [0.9,0.1]
]) 


#################################
# Learn Extra structure coordinates
#################################
learn_extra_dataA = np.array([
    [.2, 0.0], # A1
    [.2, 0.1], # A2
    [.2, 0.2], # A3
    [.2, 0.3], # A4
    [.2, 0.4], 
    [.2, 0.5], 

    [.45, 0.0], # A1
    [.45, 0.1], # A2
    [.45, 0.2], # A3
    [.45, 0.3], # A4
    [.45, 0.4],
    [.45, 0.5], 

]) 
learn_extra_dataB = np.array([
    [.55,.4], 
    [.55,.5], 
    [.55,.6], 
    [.55,.7], 
    [.55,.8], 
    [.55,.9], 

    [.8,.4], 
    [.8,.5], 
    [.8,.6], 
    [.8,.7], 
    [.8,.8], 
    [.8,.9], 
]) 
learn_extra_dataT = np.array([
    [.55,0],
    [.6,0],
    [.65,0],
    [.7,0],

    [.45,.9],
    [.4,.9],
    [.35,.9],
    [.3,.9],

    [.5,.6],
    [.5,.50],
    [.5,.40],
    [.5,.30],
]) 


###########################
#UNI V PROXY structure coordinates
########################### 
uniVproxi_dataA = np.array([
    [.9,1],
    [0,.8],
    [.9,.6],
]) 
uniVproxi_dataB = np.array([
   
    [0,.4], 
    [0,.2], 
    [0,0], 

]) 
uniVproxi_dataT = np.array([
    [.8,.4],
    [0.9,0.4],
    [1,0.4]
])



#############################################
# TOP MIDDLE BOTTOM structure coordinates
#############################################
top_middle_bottom_dataA = np.array([
    [0.4, 0.7],  # top row
    [0.5, 0.7],
    [0.9, 0.7],

    [0.5, 0.45],  # middle row
    [0.7, 0.55],
    [0.9, 0.45]
])
top_middle_bottom_dataB = np.array([
    [0.1, 0.55],  # middle row
    [0.2, 0.45],
    [0.6, 0.55],

    [0.35, 0.3],  # bottom row
    [0.5, 0.3],
    [0.6, 0.3]
])
top_middle_bottom_dataT = np.array([


    [0.1, 0.7],  # T1 (top left)
    [0.9, 0.3]   # T2 (bottom right)
])



#####################################################################################################################################
#####################################################################################################################################
##################START CHECKING DISTANCES###########################################################################################
#####################################################################################################################################
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def calculate_gcm_metrics(conditions_dict, c=1.0):
    """
    Calculates City Block and Euclidean distances, as well as Summed Similarities,
    for test points compared to Category A and Category B exemplars.
    
    Parameters:
    - conditions_dict: Dictionary mapping condition names to tuples of (dataA, dataB, dataT)
    - c: Sensitivity parameter for similarity calculation (default 1.0)
    """
    results = []
    
    for cond_name, (dataA, dataB, dataT) in conditions_dict.items():
        for i, t in enumerate(dataT):
            
            # --- City Block (L1) Calculations ---
            cb_dists_A = np.sum(np.abs(dataA - t), axis=1)
            cb_dists_B = np.sum(np.abs(dataB - t), axis=1)
            
            # Mean City Block Distance
            cb_mean_A = np.mean(cb_dists_A)
            cb_mean_B = np.mean(cb_dists_B)
            
            # Summed Similarity (City Block)
            cb_sim_A = np.sum(np.exp(-c * cb_dists_A))
            cb_sim_B = np.sum(np.exp(-c * cb_dists_B))
            
            # --- Euclidean (L2) Calculations ---
            euc_dists_A = np.sqrt(np.sum((dataA - t)**2, axis=1))
            euc_dists_B = np.sqrt(np.sum((dataB - t)**2, axis=1))
            
            # Mean Euclidean Distance
            euc_mean_A = np.mean(euc_dists_A)
            euc_mean_B = np.mean(euc_dists_B)
            
            # Summed Similarity (Euclidean)
            euc_sim_A = np.sum(np.exp(-c * euc_dists_A))
            euc_sim_B = np.sum(np.exp(-c * euc_dists_B))
            
            # Append to results
            results.append({
                'Condition': cond_name,
                'Test_Point_Index': f"T{i}",
                'Coordinates': f"({t[0]}, {t[1]})",
                
                # City Block Results
                'CB_Mean_Dist_A': round(cb_mean_A, 3),
                'CB_Mean_Dist_B': round(cb_mean_B, 3),
                'CB_Sum_Sim_A': round(cb_sim_A, 3),
                'CB_Sum_Sim_B': round(cb_sim_B, 3),
                'CB_Diff_(A-B)_Sim': round(cb_sim_A - cb_sim_B, 3),
                
                # Euclidean Results
                'Euc_Mean_Dist_A': round(euc_mean_A, 3),
                'Euc_Mean_Dist_B': round(euc_mean_B, 3),
                'Euc_Sum_Sim_A': round(euc_sim_A, 3),
                'Euc_Sum_Sim_B': round(euc_sim_B, 3),
                'Euc_Diff_(A-B)_Sim': round(euc_sim_A - euc_sim_B, 3)
            })
            
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('gcm_critical_distances.csv', index=False)
    
    return df

# Your data arrays here...
# [Insert the numpy arrays you provided here]

conditions = {
    'Full_Circle': (full_circle_dataA, full_circle_dataB, full_circle_dataT),
    'Curtailed_Circle': (curtailed_circle_dataA, curtailed_circle_dataB, curtailed_circle_dataT),
    'Learn_Extra': (learn_extra_dataA, learn_extra_dataB, learn_extra_dataT),
    'Uni_v_Proxi': (uniVproxi_dataA, uniVproxi_dataB, uniVproxi_dataT),
    'Top_Middle_Bottom': (top_middle_bottom_dataA, top_middle_bottom_dataB, top_middle_bottom_dataT)
}

# Run the function
results_df = calculate_gcm_metrics(conditions)
print(results_df.head(10)) # Preview the first few rows

def calculate_all_models(conditions_dict, standard_c=1.0, sharp_c=5.0):
    results = []
    
    for cond_name, (dataA, dataB, dataT) in conditions_dict.items():
        for i, t in enumerate(dataT):
            
            # Use City Block (L1) as established by your norming study
            dists_A = np.sum(np.abs(dataA - t), axis=1)
            dists_B = np.sum(np.abs(dataB - t), axis=1)
            
            # --- 1-Nearest Neighbor (k=1) ---
            knn1_A = np.min(dists_A)
            knn1_B = np.min(dists_B)
            
            # --- 2-Nearest Neighbor (k=2) ---
            # Sort distances and take the mean of the two smallest
            knn2_A = np.mean(np.sort(dists_A)[:2])
            knn2_B = np.mean(np.sort(dists_B)[:2])
            
            # --- GCM: Standard Sensitivity ---
            sim_std_A = np.sum(np.exp(-standard_c * dists_A))
            sim_std_B = np.sum(np.exp(-standard_c * dists_B))
            
            # --- GCM: Sharp Sensitivity ---
            sim_sharp_A = np.sum(np.exp(-sharp_c * dists_A))
            sim_sharp_B = np.sum(np.exp(-sharp_c * dists_B))
            
            results.append({
                'Condition': cond_name,
                'Test_Item': f"T{i} {tuple(t)}",
                '1-NN_Dist_A': round(knn1_A, 3),
                '1-NN_Dist_B': round(knn1_B, 3),
                '2-NN_Dist_A': round(knn2_A, 3),
                '2-NN_Dist_B': round(knn2_B, 3),
                'GCM_Std_Sim_A': round(sim_std_A, 4),
                'GCM_Std_Sim_B': round(sim_std_B, 4),
                'GCM_Sharp_Sim_A': round(sim_sharp_A, 5),
                'GCM_Sharp_Sim_B': round(sim_sharp_B, 5),
            })
            
    df = pd.DataFrame(results)
    df.to_csv('comprehensive_model_checks.csv', index=False)
    return df

# [Insert your numpy coordinate arrays here]

conditions = {
    'Full_Circle': (full_circle_dataA, full_circle_dataB, full_circle_dataT),
    'Curtailed_Circle': (curtailed_circle_dataA, curtailed_circle_dataB, curtailed_circle_dataT),
    'Learn_Extra': (learn_extra_dataA, learn_extra_dataB, learn_extra_dataT),
    'Uni_v_Proxi': (uniVproxi_dataA, uniVproxi_dataB, uniVproxi_dataT),
    'Top_Middle_Bottom': (top_middle_bottom_dataA, top_middle_bottom_dataB, top_middle_bottom_dataT)
}

# Run and display clean output for specific critical items
df = calculate_all_models(conditions, standard_c=1.0, sharp_c=5.0)

# Filter for just the Full Circle critical points to view easily in console
print("\n--- CRITICAL TEST POINTS: FULL CIRCLE ---")
critical_df = df[(df['Condition'] == 'Full_Circle') & (df['Test_Item'].str.contains('T0|T4'))]
print(critical_df[['Test_Item', '1-NN_Dist_A', '1-NN_Dist_B', 'GCM_Std_Sim_A', 'GCM_Std_Sim_B', 'GCM_Sharp_Sim_A', 'GCM_Sharp_Sim_B']].to_string(index=False))

