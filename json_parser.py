def HMD_mapper(information):
    def extract_paths(node, path, paths):
        """
        Recursively extract paths from the hierarchical VMD structure.
        """
        current_path = path + [node["parent"]]
        if "children" in node and isinstance(node["children"], list) and node["children"]:
            for child in node["children"]:
                extract_paths(child, current_path, paths)
        else:
            paths.append(current_path)

    def flatten_data(data):
        """
        Flattens the data list while ensuring correct mapping to hierarchical paths.
        """
        flat_data = []
        for row in data:
          flat_data.append(row)
        return flat_data

    def mapper(VMD, data):
        """
        Maps hierarchical VMD data to flat data list dynamically.
        """
        paths = []
        for item in VMD:
            extract_paths(item, [], paths)

        flat_data = flatten_data(data)
        mapped_data = {" -> ".join(path): datum for path, datum in zip(paths, flat_data)}

        return mapped_data

    information_to_be_checked=[]
    for table_name in information.keys():
        VMD_f=[]
        Data_f=[]
        for index in range(len(information[table_name])):
            if index+1==1: # VMD
              for VMD in information[table_name][index].values():
                VMD_f.append(VMD)
            elif index+1==2: # DATA
                for data in information[table_name][index].values():
                    Data_f.append(data)
            else:
                pass
        
        # # Skip if VMD_f or Data_f is empty
        # if not VMD_f or not Data_f:
        #     continue
            
        mapper_result=mapper(VMD_f[0],Data_f[0])

        for hierarchy,daum in mapper_result.items():
            s=f'{table_name} -> {hierarchy} -> {daum}'
            information_to_be_checked.append(s)

    return information_to_be_checked


def VMD_mapper(information):
    def extract_paths(node, path, paths):
        """
        Recursively extract paths from the hierarchical VMD structure.
        """
        current_path = path + [node["parent"]]
        if "children" in node and isinstance(node["children"], list) and node["children"]:
            for child in node["children"]:
                extract_paths(child, current_path, paths)
        else:
            paths.append(current_path)

    def flatten_data(data):
        """
        Flattens the data list while ensuring correct mapping to hierarchical paths.
        """
        curr_max=0
        for row in data:
           if len(row)>curr_max:
              curr_max=len(row)
        flat_data = [[] for _ in range(curr_max)]
        for row in data:
          for entry in range(len(row)):
              flat_data[entry].append(row[entry])
        return flat_data

    def mapper(VMD, data):
        """
        Maps hierarchical VMD data to flat data list dynamically.
        """
        paths = []
        for item in VMD:
            extract_paths(item, [], paths)

        flat_data = flatten_data(data)
        mapped_data = {" -> ".join(path): datum for path, datum in zip(paths, flat_data)}
        return mapped_data

    information_to_be_checked=[]
    for table_name in information.keys():
        VMD_f=[]
        Data_f=[]
        for index in range(len(information[table_name])):
            if index+1==1: # VMD
              for VMD in information[table_name][index].values():
                VMD_f.append(VMD)
            elif index+1==2: # DATA
                for data in information[table_name][index].values():
                    Data_f.append(data)
            else:
                pass
                
        # # Skip if VMD_f or Data_f is empty
        # if not VMD_f or not Data_f:
        #     continue
            
        mapper_result=mapper(VMD_f[0],Data_f[0])
        for hierarchy,daum in mapper_result.items():
            s=f'{table_name} -> {hierarchy} -> {daum}'
            information_to_be_checked.append(s)
            
    return information_to_be_checked
# def HMD_mapper(information):
#     def extract_paths(node, path, paths):
#         """
#         Recursively extract paths from the hierarchical VMD structure.
#         """
#         current_path = path + [node["parent"]]
#         if "children" in node and isinstance(node["children"], list) and node["children"]:
#             for child in node["children"]:
#                 extract_paths(child, current_path, paths)
#         else:
#             paths.append(current_path)

#     def flatten_data(data):
#         """
#         Flattens the data list while ensuring correct mapping to hierarchical paths.
#         """
#         flat_data = []
#         for row in data:
#           flat_data.append(row)
#         return flat_data

#     def mapper(VMD, data):
#         """
#         Maps hierarchical VMD data to flat data list dynamically.
#         """
#         paths = []
#         for item in VMD:
#             extract_paths(item, [], paths)

#         flat_data = flatten_data(data)
#         mapped_data = {" -> ".join(path): datum for path, datum in zip(paths, flat_data)}

#         return mapped_data

#     information_to_be_checked=[]
#     for table_name in information.keys():
#         VMD_f=[]
#         Data_f=[]
#         for index in range(len(information[table_name])):
#             if index+1==1: # VMD
#               for VMD in information[table_name][index].values():
#                 VMD_f.append(VMD)
#             elif index+1==2: # DATA
#                 for data in information[table_name][index].values():
#                     Data_f.append(data)
#             else:
#                 pass
        
#         # Skip if VMD_f or Data_f is empty
#         if not VMD_f or not Data_f:
#             continue
            
#         mapper_result=mapper(VMD_f[0],Data_f[0])

#         for hierarchy,daum in mapper_result.items():
#             s=f'{table_name} -> {hierarchy} -> {daum}'
#             information_to_be_checked.append(s)

#     return information_to_be_checked


# def VMD_mapper(information):
#     def extract_paths(node, path, paths):
#         """
#         Recursively extract paths from the hierarchical VMD structure.
#         """
#         current_path = path + [node["parent"]]
#         if "children" in node and isinstance(node["children"], list) and node["children"]:
#             for child in node["children"]:
#                 extract_paths(child, current_path, paths)
#         else:
#             paths.append(current_path)

#     def flatten_data(data):
#         """
#         Flattens the data list while ensuring correct mapping to hierarchical paths.
#         """
#         curr_max=0
#         for row in data:
#            if len(row)>curr_max:
#               curr_max=len(row)
#         flat_data = [[] for _ in range(curr_max)]
#         for row in data:
#           for entry in range(len(row)):
#               flat_data[entry].append(row[entry])
#         return flat_data

#     def mapper(VMD, data):
#         """
#         Maps hierarchical VMD data to flat data list dynamically.
#         """
#         paths = []
#         for item in VMD:
#             extract_paths(item, [], paths)

#         flat_data = flatten_data(data)
#         mapped_data = {" -> ".join(path): datum for path, datum in zip(paths, flat_data)}
#         return mapped_data

#     information_to_be_checked=[]
#     for table_name in information.keys():
#         VMD_f=[]
#         Data_f=[]
#         for index in range(len(information[table_name])):
#             if index+1==1: # VMD
#               for VMD in information[table_name][index].values():
#                 VMD_f.append(VMD)
#             elif index+1==2: # DATA
#                 for data in information[table_name][index].values():
#                     Data_f.append(data)
#             else:
#                 pass
                
#         # Skip if VMD_f or Data_f is empty
#         if not VMD_f or not Data_f:
#             continue
            
#         mapper_result=mapper(VMD_f[0],Data_f[0])
#         for hierarchy,daum in mapper_result.items():
#             s=f'{table_name} -> {hierarchy} -> {daum}'
#             information_to_be_checked.append(s)
            
#     return information_to_be_checked