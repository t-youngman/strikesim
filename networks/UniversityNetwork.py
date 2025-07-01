import pandas as pd
import numpy as np
import networkx as nx
import os
from collections import Counter

employer_name = "Emory"

def generate_university_network(num_workers: int = None,
                              lab_size_n: int = 3,
                              lab_size_prob: float = 0.50,
                              total_friends_n: int = 3,
                              total_friends_prob: float = 0.4,
                              lab_friends_prob: float = 0.6,
                              department_friends_prob: float = 0.3,
                              university_friends_prob: float = 0.1,
                              department_data_file: str = None,
                              num_departments: int = None,
                              avg_department_size: int = None,
                              employer_name: str = "Emory"):
    """
    Generate a university-style network using the approach from UniversityNetwork.R
    
    Args:
        num_workers: Total number of workers (if None, will be determined by department data)
        lab_size_n: Negative binomial parameter n for lab sizes
        lab_size_prob: Negative binomial parameter p for lab sizes
        total_friends_n: Negative binomial parameter n for total friends
        total_friends_prob: Negative binomial parameter p for total friends
        lab_friends_prob: Probability of connections to lab mates
        department_friends_prob: Probability of connections to department mates
        university_friends_prob: Probability of connections to university mates
        department_data_file: Path to department data CSV file
        num_departments: Number of departments to generate (if synthetic data needed)
        avg_department_size: Average department size (if synthetic data needed)
    
    Returns:
        tuple: (Student_Network, Student_Edges)
            - Student_Network: NetworkX graph object
            - Student_Edges: numpy array of edges
    """
    
    # Determine department data file path if not provided
    if department_data_file is None:
        department_data_file = f"data/{employer_name}DepartmentData.csv"
    
    # Load or generate department size data
    try:
        Department_Data = pd.read_csv(department_data_file)
        print(f"Loaded department data from {department_data_file}")
    except FileNotFoundError:
        print(f"Department data file {department_data_file} not found. Generating synthetic data.")
        # Import the department data generation function
        try:
            from generate_department_data import generate_department_data
            if num_departments is None:
                num_departments = 5
            if avg_department_size is None:
                avg_department_size = 20
            Department_Data = generate_department_data(
                num_departments=num_departments,
                avg_department_size=avg_department_size,
                output_file=department_data_file
            )
        except ImportError:
            print("Warning: Could not import generate_department_data. Creating basic synthetic data.")
            # Create basic synthetic data
            departments = [f'{employer_name} Department A', f'{employer_name} Department B', 
                         f'{employer_name} Department C', f'{employer_name} Department D', 
                         f'{employer_name} Department E']
            sizes = [20, 15, 12, 18, 16]
            Department_Data = pd.DataFrame({
                'Department': departments,
                'Size': sizes
            })
    
    # Construct data frame that holds data for each individual and assign them to a lab group.
    Students = Department_Data.copy()
    # Replicate rows based on Size column (equivalent to uncount)
    Students = Students.loc[Students.index.repeat(Students['Size'])].reset_index(drop=True)
    # Add row ID (equivalent to rowid_to_column)
    Students['ID'] = range(len(Students))
    # Add new columns (equivalent to add_column)
    Students['Lab'] = np.nan
    Students['TotalFriends'] = np.nan
    Students['LabFriends'] = np.nan
    Students['DepartmentFriends'] = np.nan
    Students['UniversityFriends'] = np.nan
    
    # If num_workers is specified, truncate or extend the student list
    if num_workers is not None:
        if len(Students) > num_workers:
            Students = Students.head(num_workers).copy()
            Students['ID'] = range(num_workers)
        elif len(Students) < num_workers:
            # Extend by duplicating some students
            extra_needed = num_workers - len(Students)
            extra_students = Students.sample(n=min(extra_needed, len(Students)), replace=True)
            extra_students['ID'] = range(len(Students), num_workers)
            Students = pd.concat([Students, extra_students], ignore_index=True)
    
    # How large do the lab groups need to be? 
    ## Assume lab groups follow negative binomial distribution, loosely based off of literature.
    Lab_Sizes = np.random.negative_binomial(n=lab_size_n, p=lab_size_prob, size=1000) + 1  # Crudely add one such that there are no labs of size 0.
    Lab_Assignments = []
    for i, size in enumerate(Lab_Sizes):
        Lab_Assignments.extend([i+1] * size)  # Lab IDs start from 1
    
    List_Departments = Students['Department'].unique()
    
    for dept in List_Departments:
        temp = Students[Students['Department'] == dept].copy()
        temp['Lab'] = Lab_Assignments[:len(temp)]
        
        Students = Students[Students['Department'] != dept]
        Students = pd.concat([Students, temp], ignore_index=True)
        
        # Remove used lab assignments
        Lab_Assignments = Lab_Assignments[len(temp):]
    
    # How to construct the network?
    ## Inelegant method: Assume that the total number of edges each agent has follows some distribution (e.g. negative binomial).
    ## Each agent will then "assign" each of those edges to their lab, their department, or the broader university.
    ## These assignations will be weighted, such that the majority (~60%) of edges are assigned to lab, fewer are assigned to department (~30%), and the least assigned to university (~10%).
    Students['TotalFriends'] = np.random.negative_binomial(n=total_friends_n, p=total_friends_prob, size=len(Students))
    
    for i in range(len(Students)):
        # Sample connection types with probabilities
        temp = np.random.choice([1, 2, 3], size=int(Students.iloc[i]['TotalFriends']), 
                               p=[lab_friends_prob, department_friends_prob, university_friends_prob], replace=True)
        temp_table = Counter(temp)
        
        Students.iloc[i, Students.columns.get_loc('LabFriends')] = temp_table.get(1, 0)
        Students.iloc[i, Students.columns.get_loc('DepartmentFriends')] = temp_table.get(2, 0)
        Students.iloc[i, Students.columns.get_loc('UniversityFriends')] = temp_table.get(3, 0)
    
    # Next need to create adjacency matrix that describes all of the contacts in the network.
    ## First create data frame object that will hold all edge information.
    max_friends = Students['TotalFriends'].max()
    Student_Adjacency = np.full((len(Students), 1 + int(max_friends)), np.nan)
    
    for i in range(len(Students)):
        Current_Student = int(Students.iloc[i]['ID'])
        Current_Lab = int(Students.iloc[i]['Lab'])
        Current_Department = Students.iloc[i]['Department']
        
        Student_Adjacency[i, 0] = Current_Student
        
        # Get lab mates (excluding current student)
        Lab_Mates_Total = Students[Students['Lab'] == Current_Lab]['ID'].tolist()
        Lab_Mates = [x for x in Lab_Mates_Total if x != Current_Student]
        
        # Get department mates (excluding lab mates)
        Department_Mates_Total = Students[Students['Department'] == Current_Department]['ID'].tolist()
        Department_Mates = [x for x in Department_Mates_Total if x not in Lab_Mates_Total]
        
        # Get university mates (excluding department mates)
        University_Mates_Total = Students['ID'].tolist()
        University_Mates = [x for x in University_Mates_Total if x not in Department_Mates_Total]
        
        ## Identify Lab contacts.
        if len(Lab_Mates) >= int(Students.iloc[i]['LabFriends']):
            Lab_Contacts = np.random.choice(Lab_Mates, size=int(Students.iloc[i]['LabFriends']), replace=False)
        else:
            Lab_Contacts = np.array(Lab_Mates)
        
        ## Identify Department contacts.
        if len(Department_Mates) >= int(Students.iloc[i]['DepartmentFriends']):
            Department_Contacts = np.random.choice(Department_Mates, size=int(Students.iloc[i]['DepartmentFriends']), replace=False)
        else:
            Department_Contacts = np.array(Department_Mates)
        
        ## Identify University contacts.
        if len(University_Mates) >= int(Students.iloc[i]['UniversityFriends']):
            University_Contacts = np.random.choice(University_Mates, size=int(Students.iloc[i]['UniversityFriends']), replace=False)
        else:
            University_Contacts = np.array(University_Mates)
        
        All_Contacts = np.concatenate([Lab_Contacts, Department_Contacts, University_Contacts])
        
        ## Add to adjacency matrix.
        if len(All_Contacts) > 0:
            temp_end_column = 1 + len(All_Contacts)
            Student_Adjacency[i, 1:temp_end_column] = All_Contacts
    
    # Need to convert to edge list.   
    temp_1 = pd.DataFrame(Student_Adjacency)
    temp_2 = temp_1.melt(id_vars=[0], var_name="Contact_Number", value_name="Contact")
    temp_2 = temp_2.drop('Contact_Number', axis=1)
    temp_2 = temp_2.dropna()
    
    Student_Edges = temp_2.values
    Student_Edges = Student_Edges.astype(int)  # Convert to integer IDs
    
    # Create NetworkX graph from edge list
    Student_Network = nx.from_edgelist(Student_Edges)
    
    return Student_Network, Student_Edges

# Return the network and edges for use in other scripts
def get_university_network(employer_name: str = "Emory"):
    """
    Returns the generated university network and edge list.
    
    Args:
        employer_name (str): Name of the employer/university
    
    Returns:
        tuple: (Student_Network, Student_Edges)
            - Student_Network: NetworkX graph object
            - Student_Edges: numpy array of edges
    """
    Student_Network, Student_Edges = generate_university_network(employer_name=employer_name)
    return Student_Network, Student_Edges

def save_network_to_csv(filename: str = None, employer_name: str = "Emory"):
    """
    Generate the network edge list and save to a CSV file.
    
    Args:
        filename (str): Path to save the CSV file (if None, will use employer_name)
        employer_name (str): Name of the employer/university
    """
    if filename is None:
        filename = f"employers/{employer_name}EdgeList.csv"
    
    Student_Network, Student_Edges = generate_university_network(employer_name=employer_name)
    Student_Edges_df = pd.DataFrame(Student_Edges, columns=['From', 'To'])
    Student_Edges_df.to_csv(filename, index=False)
    print(f"Network saved to {filename}")

def calculate_expected_workers(num_departments: int = None,
                             avg_department_size: int = None,
                             department_data_file: str = None,
                             employer_name: str = "Emory") -> int:
    """
    Calculate the expected number of workers based on department parameters
    
    Args:
        num_departments: Number of departments to generate (if synthetic data needed)
        avg_department_size: Average department size (if synthetic data needed)
        department_data_file: Path to department data CSV file
        employer_name: Name of the employer for file naming
    
    Returns:
        int: Expected number of workers
    """
    # Determine department data file path if not provided
    if department_data_file is None:
        department_data_file = f"data/{employer_name}DepartmentData.csv"
    
    # Try to load existing department data
    try:
        Department_Data = pd.read_csv(department_data_file)
        total_workers = Department_Data['Size'].sum()
        return total_workers
    except FileNotFoundError:
        # If no file exists, calculate based on synthetic parameters
        if num_departments is None:
            num_departments = 5
        if avg_department_size is None:
            avg_department_size = 20
        
        # Estimate total workers (this is approximate since actual sizes vary)
        estimated_total = num_departments * avg_department_size
        return estimated_total

# Original script functionality (for backward compatibility)
if __name__ == "__main__":
    # Generate the network edge list and save to a CSV file.
    save_network_to_csv(employer_name=employer_name)
   
    # Optional: Plot the network (equivalent to commented R code)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 8))
    # pos = nx.spring_layout(Student_Network)
    # nx.draw(Student_Network, pos,
    #         node_size=2,
    #         node_color='lightblue',
    #         edge_color='gray',
    #         width=0.25,
    #         with_labels=False)
    # plt.title("University Network")
    # plt.show()