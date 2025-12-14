import os
import xml.etree.ElementTree as ET
import re

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Resources
        res = root.find('./AreaEstimates/Resources')
        bram = res.find('BRAM_18K').text
        dsp = res.find('DSP').text
        ff = res.find('FF').text
        lut = res.find('LUT').text
        
        # Latency
        lat = root.find('./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency').text
        
        return {
            'BRAM': bram,
            'DSP': dsp, 
            'FF': ff,
            'LUT': lut,
            'Latency': lat
        }
    except Exception as e:
        return None

def check_constraints(log_path):
    status = "N/A"
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            if "All loop constraints were satisfied" in content:
                status = "Satisfied"
            elif "All loop constraints were NOT satisfied" in content:
                status = "NOT Satisfied"
    except:
        pass
    return status

base_dir = "/home/er495/smolVLA-Cornell/hardware_build/shortned_ablation_fixed"
items = sorted(os.listdir(base_dir))

print(f"{'Project':<50} | {'Lat (Cyc)':<15} | {'BRAM':<8} | {'DSP':<8} | {'Constraints':<15}")
print("-" * 110)

for item in items:
    path = os.path.join(base_dir, item)
    if os.path.isdir(path):
        xml_path = os.path.join(path, "csynth.xml")
        log_path = os.path.join(path, "solution1.log")
        
        data = parse_xml(xml_path)
        constraints = check_constraints(log_path)
        
        if data:
            print(f"{item:<50} | {data['Latency']:<15} | {data['BRAM']:<8} | {data['DSP']:<8} | {constraints:<15}")
