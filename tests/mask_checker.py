"""Runs pylint on mask.py with desired attributes"""
import subprocess, sys, os, re, argparse

def read_file_to_data(file_to_modify):
    """ """
    modifing_file = open(file_to_modify, "r")
    contents = modifing_file.readlines()
    modifing_file.close()
    return contents

def write_data_to_file(file_to_modify, write_data):
    """"""
    modifing_file = open(file_to_modify, "w")
    contents = "".join(write_data)
    modifing_file.write(contents)
    modifing_file.close()

def find_missing_docstring(lint_file, error_dict=dict()):
    """ """
    error_type = "Missing docstring"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            if m:
                int1 = m.group(1)
                leading_spaces = len(line) - len(line.lstrip()) + 4
                nex = " " * leading_spaces
                out_string = nex+'""" """\n'

                error_dict[int(int1)] = out_string

    return error_dict

def fix_missing_docstring(lint_file, contents, error_dict=dict()):
    """ """
    errors = find_missing_docstring(lint_file, error_dict)

    for key in reversed(sorted(errors)):
        contents.insert(int(key), errors[key])

    return contents

def find_operator_space(lint_file, error = list()):
    """ """
    error_type = "Operator not preceded by a space"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            if m:
                int1 = m.group(1)
                error.append(int(int1) - 1)

    return error

def fix_operator_space(lint_file, contents, error=list()):
    """ """
    error_list = find_operator_space(lint_file, error)

    for key in reversed(sorted(error_list)):
        stripped = contents[key]
        print(stripped)
        if "=" in contents[key]:
            stripped = re.sub('=', ' = ', stripped)
            contents[key] = contents[key].replace(contents[key], stripped)
            if " =" in contents[key]:
                stripped = re.sub(' =', ' =', stripped)
                contents[key] = contents[key].replace(contents[key], stripped)
            if "= " in contents[key]:
                stripped = re.sub('= ', '= ', stripped)
                contents[key] = contents[key].replace(contents[key], stripped)

    return contents

def find_dangerous_default_value(lint_file, error_dict=dict()):
    """ """
    error_type = "Dangerous default value"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            print(line)
            if m:
                int1 = m.group(1)

                re1 = '.*?' # Non-greedy match on filler
                re2 = '(\\(.*\\))' # Round Braces 1

                rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
                m = rg.search(line)
                if m:
                    out_string = m.group(1)
                    out_string = re.sub(r"\(\) \(__builtin__.", "", str(out_string))
                    out_string = re.sub(r"\)", "", out_string)
                    error_dict[int(int1) - 1] = out_string

    return error_dict

def fix_dangerous_default_value(lint_file, contents, error_dict=dict()):
    """ """
    errors = find_dangerous_default_value(lint_file, error_dict)

    for key in reversed(sorted(errors)):
        stripped = re.sub((str(errors[key])+r"\(\)"), '"None"', contents[key])
        contents[key] = contents[key].replace(contents[key], stripped)

        result = re.search(',(.*?)="None"', contents[key])
        if result == "None":
            result = re.search(r'\((.*?)="None"', contents[key])
        var_name = result.group(1)
        print(str(var_name).strip()+" "+str(key))
        if errors[key] == 'dict':
            extraline = 'if '+str(var_name)+' is "None": '+str(var_name)+' = dict()'
        else:
            extraline = 'if '+str(var_name)+' is "None": '+str(var_name)+' = []'

        leading_spaces = len(contents[key]) - len(contents[key].lstrip()) + 4
        nex = " " * leading_spaces
        out_string = nex+'"""'
        if re.match('^'+out_string, contents[key+1]):
            contents.insert(key+1, extraline)
        else:
            contents.insert(key+2, extraline)

    return contents

def find_comma_space(lint_file, error=list()):
    """ """
    error_type = "Comma not followed by a space"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            if m:
                int1 = m.group(1)
                error.append(int(int1) -1)

    return error

def fix_comma_space(lint_file, contents, error=list()):
    """ """
    error_list = find_comma_space(lint_file, error)

    for key in reversed(sorted(error_list)):
        stripped = contents[key]
        print(stripped)
        if "," in contents[key]:
            stripped = re.sub(',', ', ', stripped)
            contents[key] = contents[key].replace(contents[key], stripped)
            if ", " in contents[key]:
                stripped = re.sub(', ', ', ', stripped, count=1)
                contents[key] = contents[key].replace(contents[key], stripped)

    return contents

def find_line_length_errors(lint_file, error_list = list()):
    """ """
    error_type = "Line too long"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            if m:
                int1 = m.group(1)
                error_list.append(int(int1)-1)

    error_list.reverse()
    return error_list

def fix_line_length_errors(in_file, contents, error_list=list()):
    """ """
    errors = find_line_length_errors(in_file, error_list)

    #print errors
    for linekey in errors:
        stripped = contents[linekey]
        leading_spaces = len(stripped) - len(stripped.lstrip()) + 4
        nex = " " * leading_spaces
        if "#" in contents[linekey]:
            stripped = re.sub(r'\# ', r'\# \n'+nex, stripped, count=1)
            contents[linekey] = contents[linekey].replace(contents[linekey], stripped)
        elif "(" in contents[linekey]:
            stripped = re.sub(r'\(', r'\(\n'+nex, stripped, count=1)
            contents[linekey] = contents[linekey].replace(contents[linekey], stripped)
        elif "," in contents[linekey]:
            stripped = re.sub(r'\,', ',\n'+nex, stripped, count=1)
            contents[linekey] = contents[linekey].replace(contents[linekey], stripped)

    return contents

def find_unused_import(lint_file, error_dict=dict()):
    """ """
    error_type = "Unused import"

    for line in lint_file:
        if error_type in line:
            re1 = '.*?' # Non-greedy match on filler
            re2 = '(\\d+)' # Integer Number 1
            rg = re.compile(re1+re2, re.IGNORECASE|re.DOTALL)
            m = rg.search(line)
            if m:
                int1 = m.group(1)

                str1 = '.*?' # Non-greedy match on filler
                str2 = '(?:[a-z][a-z]+)' # Uninteresting: word
                str6 = '((?:[a-z][a-z]+))' # Word 1

                rg = re.compile(str1+str2+str1+str2+str1+str6,
                    re.IGNORECASE|re.DOTALL)
                n = rg.search(line)
                if n:
                    out_string = n.group(1)
                    error_dict[out_string] = str((int(int1) - 1))
    
    return error_dict

def fix_unused_import(in_file, contents, error_dict=dict()):
    """ """
    errors = find_unused_import(in_file, error_dict)

    for number, line in enumerate(contents):
        #print str(number)+" : "+str(line)
        stripped = contents[number]
        for key in errors:
            if str(errors[key]) == str(number):
                stripped = re.sub(key, '', stripped)
                stripped.strip()
                if stripped.strip() == "import":
                    stripped = ""
                else:
                    stripped.rstrip().rstrip(",")
                    stripped = re.sub(', ,', ',', stripped)
                    stripped = re.sub('import ,', 'import', stripped)
                contents[number] = contents[number].replace(contents[number], stripped)

    return contents

def apply_fixes(lint_file, modified_file):
    """ """
    contents = read_file_to_data(modified_file)
    lint_data = read_file_to_data(lint_file)

    contents = fix_comma_space(lint_data, contents)
    contents = fix_operator_space(lint_data, contents)
    contents = fix_unused_import(lint_data, contents)
    contents = fix_missing_docstring(lint_data, contents)
    print("attempting dangerous defaults")
    contents = fix_dangerous_default_value(lint_data, contents)
    print("dangerous defaults done")

    #contents = fix_line_length_errors(lint_data, contents)
    #contents = fix_variable_names(lint_data, contents)
    
    write_data_to_file(modified_file, contents)

def apply_lint(lint_file):
    """ """

    recorded_lint = 'temp.txt'
    if os.path.isdir(os.path.join(os.path.dirname( __file__ ), os.pardir)):
        sys.path.insert(0, os.path.join(os.path.dirname( __file__ ), os.pardir))

    subprocess.call("pylint --const-rgx='[a-z_][a-z0-9_]{2,30}$'" \
        " --disable=RP0401 --disable=RP0001 --disable=RP0002 " \
        "--disable=RP0101 --disable=RP0701 --disable=RP0801 "+lint_file+" > "+recorded_lint,
        shell=True)

    apply_fixes(recorded_lint, lint_file)
    os.remove(recorded_lint)

    subprocess.call("pylint --const-rgx='[a-z_][a-z0-9_]{2,30}$'" \
        " --disable=RP0401 --disable=RP0001 --disable=RP0002 " \
        "--disable=RP0101 --disable=RP0701 --disable=RP0801 "+lint_file,
        shell=True)

def calculate_uml(file_path, uml_name):
    """Runs UML image generator"""

    uml_path = "../docs/uml/"
    if not os.path.isdir(uml_path):
        os.makedirs(uml_path)
        
    subprocess.call("pyreverse --project="+uml_name+"UML --filter-mode=ALL "+file_path+uml_name+".py",
        shell=True)

    subprocess.call("dot -Tpng classes_"+uml_name+"UML.dot -o "+uml_name+"UML.png",
        shell=True)
    os.remove("classes_"+uml_name+"UML.dot")
    os.rename(uml_name+"UML.png", uml_path+uml_name+"_uml.png")

def main():
    """This is the main"""
    tested_file = "laugh"
    file_path = "../bin/"
    tested_file_python = file_path+tested_file+".py"

    calculate_uml(file_path, tested_file)

    subprocess.call("coverage run "+tested_file_python, shell=True)
    subprocess.call("coverage report", shell=True)

    #subprocess.call("python -m unittest test_"+tested_file+".py")
    apply_lint(tested_file_python)

if __name__ == '__main__':
    main()
