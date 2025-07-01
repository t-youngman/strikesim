library(tidyverse)
library(igraph)

setwd("../Data")

# Load department size data after processing from raw Emory data.
Department.Data <- read.csv("../DepartmentData.csv")

# Construct data frame that holds data for each individual and assign them to a lab group.
Students <- Department.Data %>%
  uncount(weights = Size) %>%
  rowid_to_column("ID") %>%
  add_column("Lab" = NA, "TotalFriends" = NA, "LabFriends" = NA,
             "DepartmentFriends" = NA, "UniversityFriends" = NA)
  
# How large do the lab groups need to be? 
## Assume lab groups follow negative binomial distribution, loosely based off of literature.
Lab.Sizes <- rnbinom(n = 1000, size = 3, prob = 0.50) + 1 # Crudely add one such that there are no labs of size 0.
Lab.Assignments <- rep(x = c(1:1000), times = Lab.Sizes)

List.Departments <- as.vector(unique(Students[,"Department"]))

for (i in List.Departments) {
  temp <- Students %>%
    filter(Department == i)
  temp[,"Lab"] <- Lab.Assignments[1:nrow(temp)]
  
  Students <- Students %>%
    filter(!Department == i) %>%
    bind_rows(temp)
  
  temp.remove <- Lab.Assignments[nrow(temp)]
  Lab.Assignments <- Lab.Assignments[-c(1:nrow(temp))]
  Lab.Assignments <- Lab.Assignments[!Lab.Assignments == temp.remove]
}
    

# How to construct the network?
## Inelegant method: Assume that the total number of edges each agent has follows some distribution (e.g. negative binomial).
## Each agent will then "assign" each of those edges to their lab, their department, or the broader university.
## These assignations will be weighted, such that the majority (~60%) of edges are assigned to lab, fewer are assigned to department (~30%), and the least assigned to university (~10%).
Students[,"TotalFriends"] <- rnbinom(n = nrow(Students), size = 3, prob = 0.4)

for (i in 1:nrow(Students)) {
  temp <- sample(x = c(1,2,3), size = as.numeric(Students[i,"TotalFriends"]), prob = c(0.6, 0.3, 0.1), replace = TRUE)
  temp.table <- table(temp)
  Students[i,"LabFriends"] <- ifelse(is.na(temp.table["1"]), 0, temp.table["1"])
  Students[i,"DepartmentFriends"] <- ifelse(is.na(temp.table["2"]), 0, temp.table["2"])
  Students[i,"UniversityFriends"] <- ifelse(is.na(temp.table["3"]), 0, temp.table["3"])
}

# Next need to create adjacency matrix that describes all of the contacts in the network.
## First create data frame object that will hold all edge information.
Student.Adjacency <- matrix(data = NA, nrow = nrow(Students), ncol = 1 + max(Students[,"TotalFriends"]), byrow = FALSE)

for (i in 1:nrow(Students)) {
  
  Current.Student <- as.numeric(Students[i,"ID"])
  Current.Lab <- as.numeric(Students[i,"Lab"])
  Current.Department <- as.character(Students[i,"Department"])
  
  Student.Adjacency[i,1] <- Current.Student
  
  Lab.Mates.Total <- as.vector(filter(Students, Lab == Current.Lab)[,"ID"])
  Lab.Mates <- Lab.Mates.Total[!Lab.Mates.Total %in% Current.Student]

  Department.Mates.Total <- as.vector(filter(Students, Department == Current.Department)[,"ID"])
  Department.Mates <- Department.Mates.Total[!Department.Mates.Total %in% Lab.Mates.Total]
  
  University.Mates.Total <- as.vector(Students[,"ID"])
  University.Mates <- University.Mates.Total[!University.Mates.Total %in% Department.Mates.Total]
  
  ## Identify Lab contacts.
  if (length(Lab.Mates) >= as.numeric(Students[i,"LabFriends"])) {
    Lab.Contacts <- sample(Lab.Mates, size = as.numeric(Students[i,"LabFriends"]), replace = FALSE)
  } else {
    Lab.Contacts <- sample(Lab.Mates, size = length(Lab.Mates), replace = FALSE)
  }
  
  ## Identify Department contacts.
  if (length(Department.Mates) >= as.numeric(Students[i,"DepartmentFriends"])) {
    Department.Contacts <- sample(Department.Mates, size = as.numeric(Students[i,"DepartmentFriends"]), replace = FALSE)
  } else {
    Department.Contacts <- sample(Department.Mates, size = length(Department.Mates), replace = FALSE)
  }
  
  ## Identify University contacts.
  if (length(University.Mates) >= as.numeric(Students[i,"UniversityFriends"])) {
    University.Contacts <- sample(University.Mates, size = as.numeric(Students[i,"UniversityFriends"]), replace = FALSE)
  } else {
    University.Contacts <- sample(University.Mates, size = length(University.Mates), replace = FALSE)
  }
  
  All.Contacts <- c(Lab.Contacts, Department.Contacts, University.Contacts)
  
  ## Add to adjacency matrix.
  if (length(All.Contacts) > 0) {
    temp.end.column <- 1 + length(All.Contacts)
    Student.Adjacency[i,2:temp.end.column] <- All.Contacts
  }
}


# Need to convert to edge list.   
temp.1 <- as_tibble(Student.Adjacency)
temp.2 <- temp.1 %>%
  pivot_longer(!V1, names_to = "Contact.Number", values_to = "Contact") %>%
  select(-Contact.Number) %>%
  filter(!is.na(Contact))

Student.Edges <- as.matrix(temp.2)
colnames(Student.Edges) <- c("From", "To")

Student.Network <- graph_from_edgelist(Student.Edges)

# plot(Student.Network,
#      vertex.size = 2,
#      vertex.label.cex = 0.1,
#      vertex.label.color = "black",
#      vertex.frame.color = "black",
#      vertex.color = "lightblue",
#      edge.color = "gray",
#      edge.width = 0.25,
#      edge.arrow.size = 0.01,
#      main = "University Network",
# )

# Get output as .csv.
# write.csv(Student.Edges, "../UniversityEdgeList.csv", row.names=FALSE)

  
  









             