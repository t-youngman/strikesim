#dependencies
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set, Tuple, Optional
import random
import os
import glob
from PIL import Image
import io


#import union and employer data, use defaults if no data provided



#generate random graphs for union and employer using the data



#specify worker agents, employer agent and union agent
class Worker:
    def __init__(self, id: int, initial_wage: float, target_wage: float, 
                 initial_savings: float = 0.0, initial_morale: float = 0.5):
        self.id = id
        self.state = 'not_striking'  # 'striking', 'not_striking', 'non_member'
        self.current_wage = initial_wage
        self.initial_wage = initial_wage
        self.target_wage = target_wage
        self.savings = initial_savings
        self.union_dues = 0.0
        self.participation_history = []  # List of daily participation decisions
        self.morale = initial_morale
        self.morale_history = [initial_morale]
        self.net_earnings = 0.0  # Cumulative earnings/losses during simulation
        self.total_expenditures = 0.0  # Track total expenditures
        
    def update_morale(self, new_morale: float):
        """Update worker's morale and add to history"""
        self.morale = max(0.0, min(1.0, new_morale))  # Clamp between 0 and 1
        self.morale_history.append(self.morale)
        
    def participate_in_strike(self, threshold: float = 0.5) -> bool:
        """Decide whether to participate in strike based on morale threshold"""
        participation = self.morale > threshold
        self.participation_history.append(participation)
        return participation
        
    def receive_wage(self, amount: float):
        """Receive wage payment"""
        self.net_earnings += amount
        
    def receive_strike_pay(self, amount: float):
        """Receive strike pay"""
        self.net_earnings += amount
        
    def pay_union_dues(self, amount: float):
        """Pay union dues"""
        self.union_dues += amount
        self.net_earnings -= amount
        
    def pay_daily_expenses(self, amount: float):
        """Pay daily living expenses"""
        self.net_earnings -= amount
        self.total_expenditures += amount

class Employer:
    def __init__(self, initial_balance: float, revenue_markup: float = 1.5,
                 concession_threshold: float = -10000.0, concession_policy: str = 'none'):
        self.balance = initial_balance
        self.revenue_markup = revenue_markup
        self.concession_threshold = concession_threshold
        self.concession_policy = concession_policy
        self.retaliation_policy = 'none'
        self.daily_revenue = 0.0
        self.total_balance = initial_balance
        self.revenue_history = []
        self.balance_history = [initial_balance]
        self.concessions_granted = 0.0
        
    def calculate_daily_revenue(self, workers_working: int, daily_wage_cost: float):
        """Calculate daily revenue based on number of workers and markup"""
        self.daily_revenue = self.revenue_markup * daily_wage_cost * workers_working
        self.revenue_history.append(self.daily_revenue)
        self.balance += self.daily_revenue
        self.total_balance = self.balance
        self.balance_history.append(self.balance)
        
    def should_grant_concession(self) -> bool:
        """Check if employer should grant concession based on threshold"""
        return self.balance < self.concession_threshold
        
    def grant_concession(self, amount: float):
        """Grant wage concession to workers"""
        self.concessions_granted += amount #is this a one-off or permanent?
        self.balance -= amount

class Union:
    def __init__(self, initial_strike_fund: float, strike_pay_rate: float = 0.5,
                 dues_rate: float = 0.02, strike_pay_policy: str = 'fixed'):
        self.strike_fund_balance = initial_strike_fund
        self.strike_pay_rate = strike_pay_rate  # As fraction of normal wage
        self.dues_rate = dues_rate  # As fraction of wage
        self.strike_pay_policy = strike_pay_policy
        self.network_policies = {
            'caucuses': False,
            'picket_lines': False,
            'strike_committees': False
        }
        self.balance_history = [initial_strike_fund]
        self.strike_pay_distributed = 0.0
        self.dues_collected = 0.0
        
    def collect_dues(self, workers: List[Worker]):
        """Collect union dues from all workers"""
        total_dues = 0.0
        for worker in workers:
            if worker.state in ['striking', 'not_striking']:  # Only members pay dues
                dues_amount = worker.current_wage * self.dues_rate
                worker.pay_union_dues(dues_amount)
                total_dues += dues_amount
        self.dues_collected += total_dues
        self.strike_fund_balance += total_dues
        self.balance_history.append(self.strike_fund_balance)
        
    def distribute_strike_pay(self, striking_workers: List[Worker]):
        """Distribute strike pay to striking workers"""
        total_strike_pay = 0.0
        for worker in striking_workers:
            if worker.state == 'striking':
                strike_pay_amount = worker.current_wage * self.strike_pay_rate
                worker.receive_strike_pay(strike_pay_amount)
                total_strike_pay += strike_pay_amount
        self.strike_pay_distributed += total_strike_pay
        self.strike_fund_balance -= total_strike_pay
        self.balance_history.append(self.strike_fund_balance)
        
    def update_policy(self, policy_name: str, value: bool):
        """Update union network policies"""
        if policy_name in self.network_policies:
            self.network_policies[policy_name] = value


#calendar initialisation

#start date
#working days


#read in initial conditions from settings.py



#initialise agents
##workers daily morale interactions



##employer daily balance interactions



##union daily balance interactions


#method to save full time series data for every agent as .h5



#method to save summary statistics as .csv


#method to run Monte Carlo simulation with different random networks



#method to save summary statistics from Monte Carlo simulation as .csv



#method to visualise the networks


#method to visualise the time series data


#methods to analyse the time series data

class StrikeSimulation:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.start_date = datetime.strptime(settings['start_date'], '%Y-%m-%d')
        self.duration = settings['duration']
        self.working_days = settings['working_days']
        
        # Initialize agents
        self.workers = []
        self.employer = None
        self.union = None
        
        # Initialize networks
        self.employer_network = None
        self.union_network = None
        
        # Simulation state
        self.current_date = self.start_date
        self.day_count = 0
        self.simulation_data = {
            'dates': [],
            'striking_workers': [],
            'working_workers': [],
            'employer_balance': [],
            'union_balance': [],
            'average_morale': [],
            'average_savings': [],
            'worker_states': []  # Track each worker's state at each timestep
        }
        
    def generate_employer_network(self, executive_size: int = 3, 
                                 department_size: int = 5, 
                                 team_size: int = 8) -> nx.Graph:
        """Generate hierarchical employer network"""
        G = nx.Graph()
        
        # Create executive level
        executives = list(range(executive_size))
        for i in executives:
            G.add_node(i, level='executive', type='executive')
        
        # Create department level
        departments = list(range(executive_size, executive_size + department_size))
        for i in departments:
            G.add_node(i, level='department', type='department')
            # Connect to executives
            for j in executives:
                G.add_edge(i, j)
        
        # Create team level
        teams = list(range(executive_size + department_size, 
                          executive_size + department_size + team_size))
        for i in teams:
            G.add_node(i, level='team', type='team')
            # Connect to departments
            for j in departments:
                G.add_edge(i, j)
        
        return G
    
    def generate_union_network(self, num_workers: int, density: float = 0.3,
                              bargaining_committee_size: int = 3,
                              team_density: float = 0.5) -> nx.Graph:
        """Generate union network with only union members as nodes, based on density."""
        G = nx.Graph()
        
        # Determine union members
        num_union_members = max(1, int(density * num_workers))
        union_member_ids = sorted(random.sample(range(num_workers), num_union_members))
        
        # Create bargaining committee (committee IDs are negative to avoid collision)
        committee = list(range(-1, -1 - bargaining_committee_size, -1))
        for i in committee:
            G.add_node(i, level='committee', type='committee')
        
        # Connect committee members to each other
        for i in committee:
            for j in committee:
                if i != j:
                    G.add_edge(i, j)
        
        # Add union members and connect to committee
        for i in union_member_ids:
            G.add_node(i, level='worker', type='worker')
            # Connect some union members to committee
            if random.random() < 0.2 and committee:
                committee_member = random.choice(committee)
                G.add_edge(i, committee_member)
            # Connect union members to each other based on team_density
            for j in union_member_ids:
                if i < j and random.random() < team_density:
                    G.add_edge(i, j)
        
        return G
    
    def load_employer_network(self, filename: str) -> nx.Graph:
        """Load employer network from .gexf or .csv file"""
        try:
            # Try to find the file with different extensions
            base_path = os.path.join('networks', 'employers', filename)
            
            # Check for .gexf file first
            gexf_path = base_path if filename.endswith('.gexf') else base_path + '.gexf'
            if os.path.exists(gexf_path):
                G = nx.read_gexf(gexf_path)
                print(f"Loaded employer network from {gexf_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                return G
            
            # Check for .csv file
            csv_path = base_path if filename.endswith('.csv') else base_path + '.csv'
            if os.path.exists(csv_path):
                # Read CSV edge list
                edges_df = pd.read_csv(csv_path)
                
                # Check if the CSV has the expected format
                if 'From' in edges_df.columns and 'To' in edges_df.columns:
                    G = nx.from_pandas_edgelist(edges_df, source='From', target='To', create_using=nx.Graph())
                    print(f"Loaded employer network from {csv_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    return G
                else:
                    raise ValueError(f"CSV file {csv_path} does not have expected 'From' and 'To' columns")
            
            # If neither file exists, raise FileNotFoundError
            raise FileNotFoundError(f"Employer network file not found: {base_path} (tried .gexf and .csv extensions)")
            
        except Exception as e:
            print(f"Error loading employer network from {filename}: {e}")
            print("Falling back to generated network")
            return self.generate_employer_network()
    
    def load_union_network(self, filename: str) -> nx.Graph:
        """Load union network from .gexf or .csv file"""
        try:
            # Try to find the file with different extensions
            base_path = os.path.join('networks', 'unions', filename)
            
            # Check for .gexf file first
            gexf_path = base_path if filename.endswith('.gexf') else base_path + '.gexf'
            if os.path.exists(gexf_path):
                G = nx.read_gexf(gexf_path)
                print(f"Loaded union network from {gexf_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                return G
            
            # Check for .csv file
            csv_path = base_path if filename.endswith('.csv') else base_path + '.csv'
            if os.path.exists(csv_path):
                # Read CSV edge list
                edges_df = pd.read_csv(csv_path)
                
                # Check if the CSV has the expected format
                if 'From' in edges_df.columns and 'To' in edges_df.columns:
                    G = nx.from_pandas_edgelist(edges_df, source='From', target='To', create_using=nx.Graph())
                    print(f"Loaded union network from {csv_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    return G
                else:
                    raise ValueError(f"CSV file {csv_path} does not have expected 'From' and 'To' columns")
            
            # If neither file exists, raise FileNotFoundError
            raise FileNotFoundError(f"Union network file not found: {base_path} (tried .gexf and .csv extensions)")
            
        except Exception as e:
            print(f"Error loading union network from {filename}: {e}")
            print("Falling back to generated network")
            return self.generate_union_network(num_workers=50)
    
    def get_available_networks(self) -> Dict[str, List[str]]:
        """Get list of available network files"""
        networks = {'employers': [], 'unions': []}
        
        # Get employer networks (both .gexf and .csv)
        employer_gexf_path = os.path.join('networks', 'employers', '*.gexf')
        employer_csv_path = os.path.join('networks', 'employers', '*.csv')
        employer_gexf_files = glob.glob(employer_gexf_path)
        employer_csv_files = glob.glob(employer_csv_path)
        
        # Combine and remove extensions
        employer_files = employer_gexf_files + employer_csv_files
        networks['employers'] = [os.path.basename(f).replace('.gexf', '').replace('.csv', '') for f in employer_files]
        
        # Get union networks (both .gexf and .csv)
        union_gexf_path = os.path.join('networks', 'unions', '*.gexf')
        union_csv_path = os.path.join('networks', 'unions', '*.csv')
        union_gexf_files = glob.glob(union_gexf_path)
        union_csv_files = glob.glob(union_csv_path)
        
        # Combine and remove extensions
        union_files = union_gexf_files + union_csv_files
        networks['unions'] = [os.path.basename(f).replace('.gexf', '').replace('.csv', '') for f in union_files]
        
        return networks
    
    def is_working_day(self, date: datetime) -> bool:
        """Check if a given date is a working day"""
        return date.strftime('%A') in self.working_days
    
    def is_strike_day(self, date: datetime) -> bool:
        """Check if a given date is a strike day based on the strike pattern"""
        strike_pattern = self.settings.get('strike_pattern', 'indefinite')
        
        if strike_pattern == 'indefinite':
            # Every working day is a potential strike day
            return self.is_working_day(date)
        
        elif strike_pattern == 'once_a_week':
            # Strike on Monday (first day of week)
            return date.strftime('%A') == 'Monday' and self.is_working_day(date)
        
        elif strike_pattern == 'once_per_month':
            # Strike on the 1st of each month
            return date.day == 1 and self.is_working_day(date)
        
        elif strike_pattern == 'weekly_escalation':
            # Escalating strike days per week
            start_day = self.settings.get('weekly_escalation_start', 1)
            days_since_start = (date - self.start_date).days
            week_number = days_since_start // 7
            strike_days_this_week = min(5, start_day + week_number)  # Cap at 5 days
            
            # Calculate which days of the week are strike days
            week_start = date - timedelta(days=date.weekday())
            strike_dates = []
            for i in range(strike_days_this_week):
                strike_date = week_start + timedelta(days=i)
                if self.is_working_day(strike_date):
                    strike_dates.append(strike_date.date())
            
            return date.date() in strike_dates
        
        else:
            # Default to indefinite
            return self.is_working_day(date)
    
    def get_next_working_day(self, date: datetime) -> datetime:
        """Get the next working day from a given date"""
        next_date = date + timedelta(days=1)
        while not self.is_working_day(next_date):
            next_date += timedelta(days=1)
        return next_date
    
    def initialize_simulation(self, num_workers: int = 50, 
                            initial_wage: float = 100.0,
                            target_wage: float = 105.0,
                            initial_employer_balance: float = 100000.0,
                            initial_strike_fund: float = 50000.0):
        """Initialize the simulation with workers, employer, and union"""
        
        # Use settings if provided, otherwise use defaults
        participation_threshold = self.settings.get('participation_threshold', 0.5)
        initial_wage = self.settings.get('initial_wage', initial_wage)
        target_wage = self.settings.get('target_wage', target_wage)
        initial_employer_balance = self.settings.get('initial_employer_balance', initial_employer_balance)
        initial_strike_fund = self.settings.get('initial_strike_fund', initial_strike_fund)
        initial_morale_range = self.settings.get('initial_morale_range', (0.5, 0.8))
        initial_savings_range = self.settings.get('initial_savings_range', (0.0, 1000.0))
        department_density = self.settings.get('department_density', 0.3)
        team_density = self.settings.get('team_density', 0.5)
        bargaining_committee_size = self.settings.get('bargaining_committee_size', 3)
        
        # Initialize networks first to determine number of workers
        employer_network_file = self.settings.get('employer_network_file', None)
        union_network_file = self.settings.get('union_network_file', None)
        employer_network_loaded = False
        union_network_loaded = False
        
        if employer_network_file:
            self.employer_network = self.load_employer_network(employer_network_file)
            employer_network_loaded = True
        else:
            self.employer_network = None
        
        if union_network_file:
            self.union_network = self.load_union_network(union_network_file)
            union_network_loaded = True
        else:
            self.union_network = None
        
        # Decide number of workers
        if employer_network_loaded and self.employer_network is not None:
            num_workers = self.employer_network.number_of_nodes()
        elif union_network_loaded and self.union_network is not None:
            num_workers = self.union_network.number_of_nodes()  # This is only for legacy support
        else:
            num_workers = self.settings.get('num_workers', num_workers)
        
        # If random networks are needed, generate them now with correct num_workers
        if self.employer_network is None:
            self.employer_network = self.generate_employer_network()
        if self.union_network is None:
            density = self.settings.get('department_density', 0.3)  # Use department_density as union density
            self.union_network = self.generate_union_network(
                num_workers=num_workers,
                density=density,
                bargaining_committee_size=bargaining_committee_size,
                team_density=team_density
            )
        
        # Create workers based on determined number
        for i in range(num_workers):
            worker = Worker(
                id=i,
                initial_wage=initial_wage,
                target_wage=target_wage,
                initial_savings=random.uniform(*initial_savings_range),
                initial_morale=random.uniform(*initial_morale_range)
            )
            self.workers.append(worker)
        
        # Create employer
        self.employer = Employer(
            initial_balance=initial_employer_balance,
            revenue_markup=self.settings.get('revenue_markup', 1.5),
            concession_threshold=self.settings.get('concession_threshold', -10000.0),
            concession_policy=self.settings.get('concession_policy', 'none')
        )
        
        # Create union
        self.union = Union(
            initial_strike_fund=initial_strike_fund,
            strike_pay_rate=self.settings.get('strike_pay_rate', 0.5),
            dues_rate=self.settings.get('dues_rate', 0.02),
            strike_pay_policy=self.settings.get('strike_pay_policy', 'fixed')
        )
        
        # Set initial strike state
        for worker in self.workers:
            if worker.morale > participation_threshold:
                worker.state = 'striking'
            else:
                worker.state = 'not_striking'

    def calculate_private_morale(self, worker: Worker, morale_spec: str = 'sigmoid') -> float:
        """Calculate worker's private morale based on financial position and target wage"""
        wage_gap = (worker.target_wage - worker.current_wage) / worker.current_wage
        savings_change = (worker.net_earnings - worker.savings) / (worker.initial_wage * max(self.day_count, 1))
        
        if morale_spec == 'sigmoid':
            # Sigmoid specification 
            def calibrate_sigmoid(reference, target):
                return -(1/reference)*np.log((1-target)/target)
            alpha = calibrate_sigmoid(self.settings.get('inflation', 0.05), 0.6)
            beta = calibrate_sigmoid(self.settings.get('belt_tightening', -0.2),0.4)
            gamma = self.settings.get('sigmoid_gamma', 1)
            # Factors
            wage_factor = 1 / (1 + np.exp(-wage_gap * alpha))  # Positive wage gap increases morale
            savings_factor = 1 / (1 + np.exp(-savings_change * beta))  # Positive savings change increases morale
            
            # Combine factors - ensure result is between 0 and 1
            combined_morale = gamma*worker.morale + (1-gamma)*(wage_factor + savings_factor)/2
            return max(0.0, min(1.0, combined_morale))
            
        elif morale_spec == 'linear':
            # Linear specification
            phi = self.settings.get('linear_phi', 0.3)
            alpha, beta, gamma = self.settings.get('linear_alpha', 0.3), self.settings.get('linear_beta', 0.3), self.settings.get('linear_gamma', 0.4)
            linear_morale = alpha * wage_gap + beta * savings_change + gamma * worker.morale
            return 1 / (1 + np.exp(-phi * linear_morale))
            
        else:  # 'no_motivation'
            # No motivation specification
            alpha, beta, gamma = self.settings.get('no_motivation_alpha', 0.5), self.settings.get('no_motivation_beta', 0.3), self.settings.get('no_motivation_gamma', 0.2)
            no_motivation_morale = alpha * wage_gap * (beta * worker.morale + gamma * savings_change)
            return max(0.0, min(1.0, no_motivation_morale))
    
    def calculate_social_morale(self, worker: Worker) -> float:
        """Calculate social morale based on network interactions"""
        if not self.union_network.has_node(worker.id):
            return 0.0
            
        # Get connected workers
        neighbors = list(self.union_network.neighbors(worker.id))
        if not neighbors:
            return 0.0
            
        # Calculate average morale of connected workers
        neighbor_morales = []
        for neighbor_id in neighbors:
            if neighbor_id < len(self.workers):
                neighbor_morales.append(self.workers[neighbor_id].morale)
        
        if neighbor_morales:
            return np.mean(neighbor_morales)
        return 0.0
    
    def update_worker_morale(self, worker: Worker, morale_spec: str = None):
        """Update worker's morale based on private and social factors"""
        if morale_spec is None:
            morale_spec = self.settings.get('morale_specification', 'sigmoid')
            
        private_morale = self.calculate_private_morale(worker, morale_spec)
        social_morale = self.calculate_social_morale(worker)
        
        # Combine private and social morale (alpha and beta weights)
        alpha = self.settings.get('private_morale_alpha', 0.9)
        beta = self.settings.get('social_morale_beta', 0.1)
        
        # If no social connections, rely more on private morale
        if social_morale == 0.0:
            alpha, beta = 0.9, 0.1
        
        new_morale = (alpha * private_morale + beta * social_morale) / (alpha + beta)
        
        # Ensure morale doesn't change too drastically in one step
        max_change = 0.1  # Maximum 10% change per day
        current_morale = worker.morale
        if abs(new_morale - current_morale) > max_change:
            if new_morale > current_morale:
                new_morale = current_morale + max_change
            else:
                new_morale = current_morale - max_change
        
        worker.update_morale(new_morale)
    
    def process_daily_financial_flows(self):
        """Process daily financial transactions"""
        # Count workers by state
        striking_workers = [w for w in self.workers if w.state == 'striking']
        working_workers = [w for w in self.workers if w.state == 'not_striking']
        
        # Pay wages to working workers - simplify this later
        daily_wage_cost = 0.0
        for worker in working_workers:
            daily_wage = worker.current_wage 
            worker.receive_wage(daily_wage)
            daily_wage_cost += daily_wage
        
        # Distribute strike pay to striking workers
        self.union.distribute_strike_pay(striking_workers)
        
        # Calculate employer revenue
        self.employer.calculate_daily_revenue(len(working_workers), daily_wage_cost)
        
        # Collect union dues daily
        self.union.collect_dues(self.workers)
        
        # Process daily living expenses for all workers
        for worker in self.workers:
            daily_expenses = worker.initial_wage * self.settings.get('daily_expenditure_rate', 0.95)
            worker.pay_daily_expenses(daily_expenses)
    
    def process_participation_decisions(self):
        """Process daily participation decisions for all workers"""
        for worker in self.workers:
            if worker.state in ['striking', 'not_striking']:  # Only members can change state
                participation = worker.participate_in_strike()
                if participation:
                    worker.state = 'striking'
                else:
                    worker.state = 'not_striking'
    
    def process_monthly_review(self):
        """Process monthly strategy review and policy changes"""
        # pending
        return 'continue'
    
    def run_daily_cycle(self):
        """Run one day of the simulation"""
        if not self.is_working_day(self.current_date):
            self.current_date += timedelta(days=1)
            return 'weekend'
        
        # Check if this is a strike day based on the pattern
        is_strike_day = self.is_strike_day(self.current_date)
        
        # Update worker morale
        for worker in self.workers:
            self.update_worker_morale(worker)
        
        # Process participation decisions only on strike days
        if is_strike_day:
            self.process_participation_decisions()
        else:
            # On non-strike days, all workers work
            for worker in self.workers:
                if worker.state in ['striking', 'not_striking']:
                    worker.state = 'not_striking'
        
        # Process financial flows
        self.process_daily_financial_flows()
        
        # Record simulation data
        striking_count = sum(1 for w in self.workers if w.state == 'striking')
        working_count = sum(1 for w in self.workers if w.state == 'not_striking')
        avg_morale = np.mean([w.morale for w in self.workers])
        avg_savings = np.mean([w.savings + w.net_earnings for w in self.workers])
        
        self.simulation_data['dates'].append(self.current_date)
        self.simulation_data['striking_workers'].append(striking_count)
        self.simulation_data['working_workers'].append(working_count)
        self.simulation_data['employer_balance'].append(self.employer.balance)
        self.simulation_data['union_balance'].append(self.union.strike_fund_balance)
        self.simulation_data['average_morale'].append(avg_morale)
        self.simulation_data['average_savings'].append(avg_savings)
        self.simulation_data['worker_states'].append([w.state for w in self.workers])
        
        # Monthly review
        if self.day_count % 20 == 0:  # Monthly review
            result = self.process_monthly_review()
            if result != 'continue':
                return result
        
        self.current_date += timedelta(days=1)
        self.day_count += 1
        return 'continue'
    
    def run_simulation(self) -> Dict:
        """Run the complete simulation"""
        self.initialize_simulation()
        
        for day in range(self.duration):
            result = self.run_daily_cycle()
            if result in ['strike_collapsed', 'employer_conceded']:
                break
        
        return self.simulation_data
    
    def save_to_hdf5(self, filename: str = 'strikesim_data.h5'):
        """Save full time series data to HDF5 format"""
        with h5py.File(filename, 'w') as f:
            # Save simulation metadata
            f.attrs['start_date'] = self.start_date.isoformat()
            f.attrs['duration'] = self.duration
            f.attrs['num_workers'] = len(self.workers)
            
            # Save time series data
            for key, values in self.simulation_data.items():
                if key == 'dates':
                    # Convert dates to strings for HDF5 storage
                    date_strings = [d.isoformat() for d in values]
                    f.create_dataset(f'time_series/{key}', data=date_strings)
                else:
                    f.create_dataset(f'time_series/{key}', data=values)
            
            # Save worker data
            worker_group = f.create_group('workers')
            for i, worker in enumerate(self.workers):
                worker_data = worker_group.create_group(f'worker_{i}')
                worker_data.attrs['id'] = worker.id
                worker_data.attrs['initial_wage'] = worker.current_wage
                worker_data.attrs['target_wage'] = worker.target_wage
                worker_data.attrs['final_state'] = worker.state
                worker_data.attrs['final_morale'] = worker.morale
                worker_data.attrs['net_earnings'] = worker.net_earnings
                worker_data.create_dataset('morale_history', data=worker.morale_history)
                worker_data.create_dataset('participation_history', data=worker.participation_history)
            
            # Save network data
            network_group = f.create_group('networks')
            # Save union network
            union_edges = list(self.union_network.edges())
            network_group.create_dataset('union_edges', data=union_edges)
            # Save employer network
            employer_edges = list(self.employer_network.edges())
            network_group.create_dataset('employer_edges', data=employer_edges)
    
    def save_summary_to_csv(self, filename: str = 'strikesim_summary.csv'):
        """Save summary statistics to CSV"""
        # Calculate summary statistics
        final_striking = self.simulation_data['striking_workers'][-1] if self.simulation_data['striking_workers'] else 0
        final_working = self.simulation_data['working_workers'][-1] if self.simulation_data['working_workers'] else 0
        final_employer_balance = self.simulation_data['employer_balance'][-1] if self.simulation_data['employer_balance'] else 0
        final_union_balance = self.simulation_data['union_balance'][-1] if self.simulation_data['union_balance'] else 0
        avg_morale = np.mean(self.simulation_data['average_morale']) if self.simulation_data['average_morale'] else 0
        avg_savings = np.mean(self.simulation_data['average_savings']) if self.simulation_data['average_savings'] else 0
        
        # Determine outcome
        if final_striking == 0:
            outcome = 'strike_collapsed'
        elif self.employer.concessions_granted > 0:
            outcome = 'employer_conceded'
        else:
            outcome = 'ongoing'
        
        summary_data = {
            'start_date': [self.start_date.isoformat()],
            'duration_days': [len(self.simulation_data['dates'])],
            'final_striking_workers': [final_striking],
            'final_working_workers': [final_working],
            'final_employer_balance': [final_employer_balance],
            'final_union_balance': [final_union_balance],
            'average_morale': [avg_morale],
            'average_savings': [avg_savings],
            'outcome': [outcome],
            'total_concessions': [self.employer.concessions_granted],
            'total_strike_pay': [self.union.strike_pay_distributed],
            'total_dues_collected': [self.union.dues_collected],
            'total_expenditures': [sum(w.total_expenditures for w in self.workers)]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        return df
    
    def run_monte_carlo(self, num_simulations: int = 100) -> pd.DataFrame:
        """Run Monte Carlo simulation with different random networks"""
        results = []
        
        for sim in range(num_simulations):
            print(f"Running simulation {sim + 1}/{num_simulations}")
            
            # Run simulation
            sim_data = self.run_simulation()
            
            # Calculate summary statistics
            final_striking = sim_data['striking_workers'][-1] if sim_data['striking_workers'] else 0
            final_working = sim_data['working_workers'][-1] if sim_data['working_workers'] else 0
            final_employer_balance = sim_data['employer_balance'][-1] if sim_data['employer_balance'] else 0
            final_union_balance = sim_data['union_balance'][-1] if sim_data['union_balance'] else 0
            avg_morale = np.mean(sim_data['average_morale']) if sim_data['average_morale'] else 0
            avg_savings = np.mean(sim_data['average_savings']) if sim_data['average_savings'] else 0
            
            # Determine outcome
            if final_striking == 0:
                outcome = 'strike_collapsed'
            elif self.employer.concessions_granted > 0:
                outcome = 'employer_conceded'
            else:
                outcome = 'ongoing'
            
            results.append({
                'simulation': sim,
                'final_striking_workers': final_striking,
                'final_working_workers': final_working,
                'final_employer_balance': final_employer_balance,
                'final_union_balance': final_union_balance,
                'average_morale': avg_morale,
                'average_savings': avg_savings,
                'outcome': outcome,
                'total_concessions': self.employer.concessions_granted,
                'total_strike_pay': self.union.strike_pay_distributed,
                'total_dues_collected': self.union.dues_collected
            })
        
        return pd.DataFrame(results)
    
    def visualize_networks(self, save_path: str = 'networks.png'):
        """Visualize the union and employer networks"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Union network
        pos_union = nx.spring_layout(self.union_network)
        nx.draw(self.union_network, pos_union, ax=ax1, 
                node_color='lightblue', node_size=100, 
                with_labels=True, font_size=8)
        ax1.set_title('Union Network')
        
        # Employer network
        pos_employer = nx.spring_layout(self.employer_network)
        nx.draw(self.employer_network, pos_employer, ax=ax2,
                node_color='lightgreen', node_size=100,
                with_labels=True, font_size=8)
        ax2.set_title('Employer Network')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_time_series(self, save_path: str = 'time_series.png'):
        """Visualize time series data"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Create day numbers for x-axis
        days = list(range(len(self.simulation_data['striking_workers'])))
        
        # Worker participation over time
        axes[0, 0].plot(days, self.simulation_data['striking_workers'], label='Striking', color='red')
        axes[0, 0].plot(days, self.simulation_data['working_workers'], label='Working', color='blue')
        axes[0, 0].set_title('Worker Participation Over Time')
        axes[0, 0].set_xlabel('Day of Simulation')
        axes[0, 0].set_ylabel('Number of Workers')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average morale over time
        axes[0, 1].plot(days, self.simulation_data['average_morale'], color='purple')
        axes[0, 1].set_title('Average Worker Morale Over Time')
        axes[0, 1].set_xlabel('Day of Simulation')
        axes[0, 1].set_ylabel('Average Morale')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average savings over time
        axes[0, 2].plot(days, self.simulation_data['average_savings'], color='brown')
        axes[0, 2].set_title('Average Worker Savings Over Time')
        axes[0, 2].set_xlabel('Day of Simulation')
        axes[0, 2].set_ylabel('Average Savings ($)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Employer balance over time
        axes[1, 0].plot(days, self.simulation_data['employer_balance'], color='green')
        axes[1, 0].set_title('Employer Balance Over Time')
        axes[1, 0].set_xlabel('Day of Simulation')
        axes[1, 0].set_ylabel('Balance ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Union balance over time
        axes[1, 1].plot(days, self.simulation_data['union_balance'], color='orange')
        axes[1, 1].set_title('Union Strike Fund Balance Over Time')
        axes[1, 1].set_xlabel('Day of Simulation')
        axes[1, 1].set_ylabel('Balance ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Hide the last subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self) -> Dict:
        """Analyze simulation results and return key statistics"""
        if not self.simulation_data['dates']:
            return {}
        
        analysis = {
            'total_days': len(self.simulation_data['dates']),
            'final_striking_workers': self.simulation_data['striking_workers'][-1],
            'final_working_workers': self.simulation_data['working_workers'][-1],
            'max_striking_workers': max(self.simulation_data['striking_workers']),
            'min_striking_workers': min(self.simulation_data['striking_workers']),
            'final_employer_balance': self.simulation_data['employer_balance'][-1],
            'final_union_balance': self.simulation_data['union_balance'][-1],
            'average_morale': np.mean(self.simulation_data['average_morale']),
            'morale_volatility': np.std(self.simulation_data['average_morale']),
            'average_savings': np.mean(self.simulation_data['average_savings']),
            'total_concessions': self.employer.concessions_granted,
            'total_strike_pay': self.union.strike_pay_distributed,
            'total_dues_collected': self.union.dues_collected,
            'total_expenditures': sum(w.total_expenditures for w in self.workers)
        }
        
        # Determine outcome
        if analysis['final_striking_workers'] == 0:
            analysis['outcome'] = 'strike_collapsed'
        elif analysis['total_concessions'] > 0:
            analysis['outcome'] = 'employer_conceded'
        else:
            analysis['outcome'] = 'ongoing'
        
        return analysis
    
    def visualize_networks_at_timestep(self, timestep: int, save_path: str = None):
        """Visualize the union and employer networks at a specific timestep"""
        if timestep >= len(self.simulation_data['worker_states']):
            return None
            
        worker_states = self.simulation_data['worker_states'][timestep]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Union network
        pos_union = nx.spring_layout(self.union_network, k=1, iterations=50)
        
        # Color nodes based on worker states
        node_colors_union = []
        node_sizes_union = []
        node_labels_union = {}
        
        for node in self.union_network.nodes():
            if node < len(worker_states):  # Worker node
                if worker_states[node] == 'striking':
                    node_colors_union.append('#ff4444')  # Bright red
                elif worker_states[node] == 'not_striking':
                    node_colors_union.append('#4444ff')  # Bright blue
                else:
                    node_colors_union.append('#888888')  # Gray
                node_sizes_union.append(400)
                node_labels_union[node] = f'W{node}'
            else:  # Committee node
                node_colors_union.append('#ffaa00')  # Orange
                node_sizes_union.append(600)
                node_labels_union[node] = f'C{abs(node)}'
        
        nx.draw(self.union_network, pos_union, ax=ax1,
                node_color=node_colors_union, node_size=node_sizes_union,
                labels=node_labels_union, font_size=10, font_weight='bold',
                edge_color='gray', alpha=0.7, width=2)
        ax1.set_title(f'Union Network - Day {timestep}', fontsize=14, fontweight='bold')
        
        # Add legend for union network
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                      markersize=15, label='Striking Worker'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', 
                      markersize=15, label='Working Worker'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaa00', 
                      markersize=15, label='Union Committee')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Employer network
        pos_employer = nx.spring_layout(self.employer_network, k=1, iterations=50)
        
        # Color nodes based on worker states
        node_colors_employer = []
        node_sizes_employer = []
        node_labels_employer = {}
        
        for node in self.employer_network.nodes():
            if node < len(worker_states):  # Worker node
                if worker_states[node] == 'striking':
                    node_colors_employer.append('#ff4444')  # Bright red
                elif worker_states[node] == 'not_striking':
                    node_colors_employer.append('#4444ff')  # Bright blue
                else:
                    node_colors_employer.append('#888888')  # Gray
                node_sizes_employer.append(400)
                node_labels_employer[node] = f'W{node}'
            else:  # Management node
                node_colors_employer.append('#44ff44')  # Bright green
                node_sizes_employer.append(600)
                node_labels_employer[node] = f'M{node}'
        
        nx.draw(self.employer_network, pos_employer, ax=ax2,
                node_color=node_colors_employer, node_size=node_sizes_employer,
                labels=node_labels_employer, font_size=10, font_weight='bold',
                edge_color='gray', alpha=0.7, width=2)
        ax2.set_title(f'Employer Network - Day {timestep}', fontsize=14, fontweight='bold')
        
        # Add legend for employer network
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                      markersize=15, label='Striking Worker'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', 
                      markersize=15, label='Working Worker'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44', 
                      markersize=15, label='Management')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_network_animation(self, save_path: str = 'network_animation.gif', 
                                fps: int = 2, max_frames: int = None):
        """Create an animated GIF showing network evolution over time"""
        if not self.simulation_data['worker_states']:
            print("No simulation data available for animation")
            return None
            
        # Determine number of frames
        total_frames = len(self.simulation_data['worker_states'])
        if max_frames and max_frames < total_frames:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            frame_indices = range(total_frames)
        
        images = []
        
        print(f"Creating animation with {len(frame_indices)} frames...")
        
        for i, timestep in enumerate(frame_indices):
            print(f"Processing frame {i+1}/{len(frame_indices)} (day {timestep})")
            
            # Generate network visualization for this timestep
            fig = self.visualize_networks_at_timestep(timestep)
            if fig is None:
                continue
                
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
            
            # Close the figure to free memory
            plt.close(fig)
        
        if not images:
            print("No images generated for animation")
            return None
            
        # Save as GIF
        print(f"Saving animation to {save_path}...")
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=1000//fps,  # Duration in milliseconds
            loop=0  # Loop indefinitely
        )
        
        print(f"Animation saved successfully! {len(images)} frames at {fps} FPS")
        return save_path

# Standalone function to create animations from saved simulation data
def create_animation_from_saved_data(hdf5_file: str, output_gif: str = 'network_animation.gif', 
                                   fps: int = 2, max_frames: int = None):
    """Create a network animation from saved simulation data"""
    try:
        # Load simulation data
        with h5py.File(hdf5_file, 'r') as f:
            # Create a minimal simulation object with loaded data
            sim = StrikeSimulation({})
            sim.simulation_data = {
                'dates': [],
                'striking_workers': [],
                'working_workers': [],
                'employer_balance': [],
                'union_balance': [],
                'average_morale': [],
                'average_savings': [],
                'worker_states': []
            }
            
            # Load time series data
            for key in sim.simulation_data.keys():
                if key == 'dates':
                    date_strings = f[f'time_series/{key}'][:]
                    sim.simulation_data[key] = [datetime.fromisoformat(d.decode()) for d in date_strings]
                else:
                    sim.simulation_data[key] = list(f[f'time_series/{key}'][:])
            
            # Load networks (simplified - would need more complex loading for full functionality)
            print("Note: Network animation from saved data may not show full network structure")
            
        # Create animation
        return sim.create_network_animation(output_gif, fps, max_frames)
        
    except Exception as e:
        print(f"Error creating animation from saved data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    print("StrikeSim Network Animation Tool")
    print("Use the dashboard to create animations interactively, or call create_animation_from_saved_data()")
