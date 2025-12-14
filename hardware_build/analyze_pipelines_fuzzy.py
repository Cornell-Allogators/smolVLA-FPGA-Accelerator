import os
import xml.etree.ElementTree as ET

def find_xml(directory, part_of_name):
    for f in os.listdir(directory):
        if part_of_name in f and f.endswith(".xml"):
            return os.path.join(directory, f)
    return None

def get_xml_data(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Try to find pipeline info
        pipeline_ii = "N/A"
        pipeline_depth = "N/A"
        
        overall = root.find('./PerformanceEstimates/SummaryOfOverallLatency')
        if overall is not None:
            ii = overall.find('PipelineInitiationInterval')
            depth = overall.find('PipelineDepth')
            if ii is not None: pipeline_ii = ii.text
            if depth is not None: pipeline_depth = depth.text

        # 2. Check SummaryOfLoopLatency (for loop-based pipelines)
        if pipeline_ii == "N/A":
            loops = root.find('./PerformanceEstimates/SummaryOfLoopLatency')
            if loops is not None:
                # Find any child that has PipelineII
                for loop in loops:
                    ii = loop.find('PipelineII')
                    depth = loop.find('PipelineDepth')
                    if ii is not None: 
                        pipeline_ii = ii.text
                        pipeline_depth = depth.text
                        break # Assume main loop is the one we want

        # Get Latency
        latency = "N/A"
        lat_node = overall.find('Average-caseLatency') if overall is not None else None
        if lat_node is not None:
            latency = lat_node.text

        return {
            'II': pipeline_ii,
            'Depth': pipeline_depth,
            'Latency': latency
        }
    except Exception as e:
        return {'Error': str(e)}

base_dir = "/home/er495/smolVLA-Cornell/hardware_build/shortned_build_importat_one_unignored"
p4_dir = os.path.join(base_dir, "final_result_dataflow_True_P_4_int8_8.prj")
p8_dir = os.path.join(base_dir, "final_result_dataflow_True_P_8_int8_16.prj")

# Map of Kernel Name -> Unique substring
kernels = {
    "Attention Loop": "Pipeline_l_attn",
    "Softmax Loop": "Pipeline_l_ex",
    "Output Loop": "l_out_row_loop_j_out",
    "Precalc Loop": "k_precalc"
}

print(f"{'Kernel':<20} | {'Metric':<10} | {'P=4':<10} | {'P=8':<10} | {'Delta':<10}")
print("-" * 70)

for user_name, pattern in kernels.items():
    p4_path = find_xml(p4_dir, pattern)
    p8_path = find_xml(p8_dir, pattern)
    
    p4_data = get_xml_data(p4_path)
    p8_data = get_xml_data(p8_path)
    
    # Just print whatever we find
    lat4 = p4_data.get('Latency', 'N/A') if p4_data else "Not Found"
    lat8 = p8_data.get('Latency', 'N/A') if p8_data else "Not Found"

    print(f"{user_name:<20} | {'Latency':<10} | {lat4:<10} | {lat8:<10} |")
    
    if p4_data:
        print(f"{'':<20} | {'II':<10} | {p4_data.get('II', 'N/A'):<10} | {p8_data.get('II', 'N/A') if p8_data else '-':<10} |")
        print(f"{'':<20} | {'Depth':<10} | {p4_data.get('Depth', 'N/A'):<10} | {p8_data.get('Depth', 'N/A') if p8_data else '-':<10} |")
    print("-" * 70)
