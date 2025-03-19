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
      if len(paths) != len(flat_data):
          pass
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
      # print(VMD_f)
      # print(Data_f)
      mapper_result=mapper(VMD_f[0],Data_f[0])

      for hierarchy,daum in mapper_result.items():
          s=f'{table_name} -> {hierarchy} -> {daum}'
          information_to_be_checked.append(s)
          # print(hierarchy)
          # print(daum)
        #   Non_elements=''
        #   for element in daum:

        #     if element in hierarchy:
        #       pass
        #     else:
        #       Non_elements+=element
        #   data_without_list=Non_elements
        #   s=f'{table_name} -> {hierarchy} -> {data_without_list}'
        #   information_to_be_checked.append(s)
    #   print(information_to_be_checked)
      print("---TABLE-----PARSED---SUCCESSFULLY----EOT---------------------")
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
      # print('Im inside the flatten_data function')
      # print(data)
      curr_max=0
      for row in data:
         if len(row)>curr_max:
            curr_max=len(row)
      flat_data = [[] for _ in range(curr_max)]
    #   print('The maximum no of columns',curr_max)
    #   print('Before entering',flat_data)
      for row in data:
        for entry in range(len(row)):
            flat_data[entry].append(row[entry])

      # print("The transformed form here is ")
      # print(flat_data)   # This whole function is just a bogus its doing nothing
      return flat_data

  def mapper(VMD, data):
      """
      Maps hierarchical VMD data to flat data list dynamically.
      """
      paths = []
      for item in VMD:
          extract_paths(item, [], paths)

      flat_data = flatten_data(data)
      if len(paths) != len(flat_data):
          pass
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
      mapper_result=mapper(VMD_f[0],Data_f[0])
      for hierarchy,daum in mapper_result.items():
          s=f'{table_name} -> {hierarchy} -> {daum}'
          information_to_be_checked.append(s)
    #   print(information_to_be_checked)
      print("----TABLE---PARSED---SUCCESSFULLY--------EOT---------------------")
  return information_to_be_checked

# print(VMD_mapper(information))