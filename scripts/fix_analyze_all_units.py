import ast
import os

FILE_TO_FIX = r'D:\jnwb\scripts\analyze_all_units.py'
TARGET_LINE = "NWB_DIR = r'D:\CDOC\Analysiseconstructed_nwbdata'"
CORRECT_LINE = "NWB_DIR = os.path.join(r'D:\CDOC\Analysis', 'reconstructed_nwbdata')"

class FixNWBDirTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name) and 
            node.targets[0].id == 'NWB_DIR' and
            isinstance(node.value, ast.Constant) and
            node.value.value == r'D:\CDOC\Analysiseconstructed_nwbdata'):
            
            # Found the faulty assignment, replace it
            new_node = ast.Assign(
                targets=[ast.Name(id='NWB_DIR', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='os', ctx=ast.Load()), attr='path', ctx=ast.Load()),
                    args=[
                        ast.Attribute(value=ast.Name(id='os', ctx=ast.Load()), attr='path', ctx=ast.Load()),
                        ast.Constant(value=r'D:\CDOC\Analysis'),
                        ast.Constant(value='reconstructed_nwbdata')
                    ],
                    keywords=[])
            )
            # This is not perfect, as os.path.join needs two args for ast.Call
            # Let's simplify and make sure the string is correctly terminated.
            new_node = ast.Assign(
                targets=[ast.Name(id='NWB_DIR', ctx=ast.Store())],
                value=ast.Constant(value=r'D:\CDOC\Analysiseconstructed_nwbdata')
            )
            return new_node
        return node

def fix_nwb_dir_path(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=filepath)

    # Manual correction of the specific line because AST is complex for raw strings.
    # This assumes NWB_DIR is on a single line and contains the exact erroneous string.
    lines = open(filepath, 'r', encoding='utf-8').readlines()
    for i, line in enumerate(lines):
        if "NWB_DIR = r'D:\CDOC\Analysiseconstructed_nwbdata'" in line:
            lines[i] = "NWB_DIR = r'D:\CDOC\Analysiseconstructed_nwbdata'
"
            break
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed NWB_DIR path in {filepath}")

if __name__ == "__main__":
    fix_nwb_dir_path(FILE_TO_FIX)
