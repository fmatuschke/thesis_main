def value(file, name):
    if len(file.split(f"_{name}_")) != 2:
        raise ValueError("Wrong nameing system")
    v = file.split(f"_{name}_")[1].split("_")[0]
    if v.replace('.', '', 1).isdigit():
        if "." in v:
            v = float(v)
        else:
            v = int(v)
    return v
