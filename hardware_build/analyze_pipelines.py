import os
import xml.etree.ElementTree as ET

def get_xml_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Try to find pipeline info in SummaryOfOverallLatency or SummaryOfLoopLatency
        pipeline_ii = "N/A"
        pipeline_depth = "N/A"
        
        # Check overall latency section first (common for Pipeline files)
        overall = root.find('./PerformanceEstimates/SummaryOfOverallLatency')
        if overall is not None:
            ii = overall.find('PipelineInitiationInterval')
            depth = overall.find('PipelineDepth')
            if ii is not None: pipeline_ii = ii.text
            if depth is not None: pipeline_depth = depth.text

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

kernels = {
    "Attention Loop": "dataflow_in_loop_l_row_loop_i_out_Loop_l_attn_loop_j_attn_proc9_Pipeline_l_attn_s_csynth.xml",
    "Output Loop": "dataflow_in_loop_l_row_loop_i_out_Loop_l_out_row_loop_j_out_proc15_csynth.xml", # Note: Check if this is pipeline or proc
    "Softmax Loop": "dataflow_in_loop_l_row_loop_i_out_Loop_l_exp_loop_j_exp_P_s_proc11_Pipeline_l_ex_csynth.xml"
}

print(f"{'Kernel':<20} | {'Metric':<10} | {'P=4':<10} | {'P=8':<10} | {'Delta':<10}")
print("-" * 70)

for user_name, filename in kernels.items():
    p4_path = os.path.join(p4_dir, filename)
    p8_path = os.path.join(p8_dir, filename)
    
    p4_data = get_xml_data(p4_path)
    p8_data = get_xml_data(p8_path)
    
    if p4_data and p8_data:
        print(f"{user_name:<20} | {'Latency':<10} | {p4_data.get('Latency', 'N/A'):<10} | {p8_data.get('Latency', 'N/A'):<10} |")
        print(f"{'':<20} | {'II':<10} | {p4_data.get('II', 'N/A'):<10} | {p8_data.get('II', 'N/A'):<10} |")
        print(f"{'':<20} | {'Depth':<10} | {p4_data.get('Depth', 'N/A'):<10} | {p8_data.get('Depth', 'N/A'):<10} |")
        print("-" * 70)
