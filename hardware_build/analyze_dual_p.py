import os
import xml.etree.ElementTree as ET

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Resources
        res = root.find('./AreaEstimates/Resources')
        dsp = int(res.find('DSP').text)
        bram = int(res.find('BRAM_18K').text)
        # Latency
        lat = int(root.find('./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency').text)
        return {'Latency': lat, 'DSP': dsp, 'BRAM': bram}
    except:
        return None

base_dir = "/home/er495/smolVLA-Cornell/hardware_build/shortned_build_importat_one_unignored"

# P_factors to analyze
p_vals = [1, 2, 4, 8]

print(f"{'P':<5} | {'P2':<5} | {'Total Unroll':<12} | {'Latency':<10} | {'DSP':<6} | {'Speedup?':<10}")
print("-" * 65)

for p in p_vals:
    # P2=1 case: Suffix is P*1
    suffix_1 = p * 1
    dir_1 = f"final_result_dataflow_True_P_{p}_int8_{suffix_1}.prj"
    
    # P2=2 case: Suffix is P*2
    suffix_2 = p * 2
    dir_2 = f"final_result_dataflow_True_P_{p}_int8_{suffix_2}.prj"
    
    path_1 = os.path.join(base_dir, dir_1, "csynth.xml")
    path_2 = os.path.join(base_dir, dir_2, "csynth.xml")
    
    data_1 = parse_xml(path_1)
    data_2 = parse_xml(path_2)
    
    if data_1:
        print(f"{p:<5} | {'1':<5} | {suffix_1:<12} | {data_1['Latency']:<10} | {data_1['DSP']:<6} | {'Baseline':<10}")
    
    if data_2:
        speedup = "N/A"
        if data_1:
            ratio = data_1['Latency'] / data_2['Latency']
            speedup = f"{ratio:.2f}x"
        print(f"{p:<5} | {'2':<5} | {suffix_2:<12} | {data_2['Latency']:<10} | {data_2['DSP']:<6} | {speedup:<10}")
    
    print("-" * 65)
