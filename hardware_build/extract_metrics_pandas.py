import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import matplotlib.pyplot as plt


def parse_csynth(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = {}
    
    # Metadata from Path
    # Expected path: .../report_NAME_DATE/csynth.xml
    dirname = os.path.basename(os.path.dirname(xml_path))
    data['Run_ID'] = dirname
    
    # Try to parse config from name (e.g., report_attention_dataflow_True_rp_8...)
    parts = dirname.split('_')
    # Default values
    data['Dataflow'] = 'Unknown'
    data['P'] = 1
    data['P_Suffix'] = 1
    
    # Heuristic parsing
    if 'dataflow_True' in dirname:
        data['Dataflow'] = True
    elif 'dataflow_False' in dirname:
        data['Dataflow'] = False
        
    # Find rp_{N} or P_{N} and int8_{M}
    for i, part in enumerate(parts):
        if part == 'rp' or part == 'P':
            if i + 1 < len(parts) and parts[i+1].isdigit():
                data['P'] = int(parts[i+1])
        if part == 'int8':
             if i + 1 < len(parts):
                val_str = parts[i+1].split('.')[0] # Strip .prj
                if val_str.isdigit():
                    data['P_Suffix'] = int(val_str)
                
    # Latency
    perf = root.find('PerformanceEstimates')
    summary = perf.find('SummaryOfOverallLatency')
    data['Latency_Cycles'] = int(summary.find('Average-caseLatency').text)
    # Clock period
    clk_est = perf.find('SummaryOfTimingAnalysis/EstimatedClockPeriod').text
    if clk_est:
        data['Clock_Period_ns'] = float(clk_est)
        data['Latency_ms'] = (data['Latency_Cycles'] * data['Clock_Period_ns']) / 1e6
        
    # Resources
    area = root.find('AreaEstimates/Resources')
    data['BRAM'] = int(area.find('BRAM_18K').text)
    data['DSP'] = int(area.find('DSP').text) if area.find('DSP') is not None else int(area.find('DSP48E').text)
    data['FF'] = int(area.find('FF').text)
    data['LUT'] = int(area.find('LUT').text)
    
    # Util Percent (if available)
    avail = root.find('AreaEstimates/AvailableResources')
    if avail is not None:
        data['BRAM_Util'] = data['BRAM'] / int(avail.find('BRAM_18K').text) * 100
        data['DSP_Util'] = data['DSP'] / int(avail.find('DSP').text) * 100
    
    return data

def main():
    root_dir = "/home/er495/smolVLA-Cornell/hardware_build/shortned_ablation_fixed"
    xml_files = glob.glob(os.path.join(root_dir, "**", "self_attention_csynth.xml"), recursive=True)
    
    records = []
    for f in xml_files:
        try:
            print(f"Parsing {f}")
            records.append(parse_csynth(f))
        except Exception as e:
            print(f"Failed to parse {f}: {e}")
            
    df = pd.DataFrame(records)
    print("Parsed Data Preview:")
    print(df.head())
    
    # Cleaning
    df = df.sort_values(by=['Dataflow', 'P', 'P_Suffix'])
    
    # Save CSV
    df.to_csv("hls_metrics.csv", index=False)
    print("Saved hls_metrics.csv")
    
    # --- Generate Powerful SVGs (Matplotlib) ---
    plt.style.use('ggplot') # Use a nice style
    
    # Convert to Millions for plotting clarity
    df['Latency_M'] = df['Latency_Cycles'] / 1e6
    
    # Plot 1: Latency Scaling (Bar)
    plt.figure(figsize=(10, 6))
    
    # Create valid label for grouping
    # We want X-axis = P_Suffix. Hue = P.
    
    df_true = df[df['Dataflow'] == True]
    
    try:
        # Aggregating duplicates by taking the minimum latency
        pivot_df = df_true.groupby(['P_Suffix', 'P'])['Latency_M'].min().unstack()
        pivot_df.plot(kind='bar', figsize=(12,7), colormap='viridis', edgecolor='black', logy=True)
        plt.legend(title='Parallelism (P)')
    except Exception as e:
        print(f"Pivot failed: {e}")
        plt.scatter(df_true['P_Suffix'], df_true['Latency_M'], c='blue')
        
    plt.title("Latency Scaling by Suffix (Log Scale) [Dataflow=True]", fontsize=16)
    plt.ylabel("Latency (Million Cycles) - Log Scale", fontsize=14)
    plt.xlabel("Suffix Factor (Int8_N)", fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', which="both", linestyle='--', alpha=0.5)
    
    # Refine Log Ticks to be readable numbers (e.g. 5, 10, 20) instead of 10^1
    from matplotlib.ticker import ScalarFormatter
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig("latency_scaling_pandas.svg", format='svg')
    plt.close()
    
    plt.tight_layout()
    plt.savefig("latency_scaling_pandas.svg", format='svg')
    plt.close()
    
    # --- Plot 2: Multi-Resource Grid (2x2) ---
    # User wants to see DSP, BRAM, LUT, FF.
    # User wants 16 points (8 Dataflow=True, 8 Dataflow=False).
    # Encoding: Color=Dataflow, Marker=P, Text=Suffix.
    
    resources = [
        ('DSP', 'DSP Usage'),
        ('BRAM', 'BRAM Usage'),
        ('LUT', 'LUT Usage'),
        ('FF', 'FF Usage')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Common markers for P factors
    markers = {1: 'o', 2: 's', 4: '^', 8: 'D', 16: 'P'}
    # Colors for Dataflow
    colors = {True: 'tab:blue', False: 'tab:red'}
    
    # Aggregate to ensure unique points (taking min latency for dupes)
    # Group by [Dataflow, P, P_Suffix]
    agg_df = df.groupby(['Dataflow', 'P', 'P_Suffix']).agg({
        'Latency_M': 'min',
        'DSP': 'mean',
        'BRAM': 'mean',
        'LUT': 'mean',
        'FF': 'mean'
    }).reset_index()
    
    print(f"Aggregated Data Points: {len(agg_df)} (Expected ~16)")
    
    for idx, (res_col, res_label) in enumerate(resources):
        ax = axes[idx]
        
        for df_val in [True, False]:
            subset = agg_df[agg_df['Dataflow'] == df_val]
            color = colors.get(df_val, 'gray')
            label_prefix = f"DF={df_val}"
            
            for p_val in sorted(subset['P'].unique()):
                p_subset = subset[subset['P'] == p_val]
                marker = markers.get(p_val, 'x')
                
                # Scatter plot
                # Label only once per group for legend cleanliness
                # But actual legend is tough with this many combos.
                # We will handle legend separately.
                
                ax.scatter(p_subset[res_col], p_subset['Latency_M'],
                          c=color, marker=marker, s=150, alpha=0.8, edgecolors='k')
                
                # Annotate with Suffix
                for _, row in p_subset.iterrows():
                    ax.annotate(f"S={row['P_Suffix']}", (row[res_col], row['Latency_M']),
                               xytext=(3, 3), textcoords='offset points', fontsize=9)

        ax.set_title(f"Latency vs {res_label}", fontsize=14)
        ax.set_xlabel(f"{res_label} (Linear)", fontsize=12)
        ax.set_ylabel("Latency (Million Cycles) (Log)", fontsize=12)
        ax.set_yscale('log')
        # Scalar Formatter for Log Y
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which="both", linestyle='--', alpha=0.5)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor='tab:blue', label='Dataflow=True', markersize=10),
        Line2D([0], [0], color='w', marker='o', markerfacecolor='tab:red', label='Dataflow=False', markersize=10),
        Line2D([0], [0], color='k', label='--- P Factor ---'),
    ]
    for p, m in markers.items():
        if p in agg_df['P'].unique():
            legend_elements.append(Line2D([0], [0], color='w', marker=m, markerfacecolor='gray', label=f'P={p}', markersize=10))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=7, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("resource_grid.svg", format='svg', bbox_inches='tight')
    plt.close()
    
    print("Generated SVGs: latency_scaling_pandas.svg, resource_grid.svg")

if __name__ == "__main__":
    main()
