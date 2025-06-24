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


#import union and employer data, use defaults if no data provided



#generate random graphs for union and employer using the data



#specify worker agents, employer agent and union agent
class Worker:
    def __init__(self, id: int, initial_wage: float, target_wage: float, 
                 initial_savings: float = 0.0, initial_morale: float = 0.5):
        self.id = id
        self.state = 'not_striking'  # 'striking', 'not_striking', 'non_member'
        self.current_wage = initial_wage
        self.target_wage = target_wage
        self.savings = initial_savings
        self.union_dues = 0.0
        self.participation_history = []  # List of daily participation decisions
        self.morale = initial_morale
        self.morale_history = [initial_morale]
        self.net_earnings = 0.0  # Cumulative earnings/losses during simulation
        
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
        self.concessions_granted += amount
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
            'average_morale': []
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
    
    def generate_union_network(self, bargaining_committee_size: int = 3,
                              department_density: float = 0.3,
                              team_density: float = 0.5) -> nx.Graph:
        """Generate union network with bargaining committee and department/team connections"""
        G = nx.Graph()
        
        # Create bargaining committee
        committee = list(range(bargaining_committee_size))
        for i in committee:
            G.add_node(i, level='committee', type='committee')
        
        # Connect committee members to each other
        for i in committee:
            for j in committee:
                if i != j:
                    G.add_edge(i, j)
        
        # Add department and team connections based on density
        # This is a simplified version - in practice, you'd want to match
        # union structure to actual workplace structure
        total_workers = len(self.workers)
        
        for i in range(total_workers):
            G.add_node(i, level='worker', type='worker')
            
            # Connect some workers to committee
            if random.random() < 0.2:  # 20% chance
                committee_member = random.choice(committee)
                G.add_edge(i, committee_member)
            
            # Connect workers to each other based on density
            for j in range(i + 1, total_workers):
                if random.random() < team_density:
                    G.add_edge(i, j)
        
        return G
    
    def is_working_day(self, date: datetime) -> bool:
        """Check if a given date is a working day"""
        return date.strftime('%A') in self.working_days
    
    def get_next_working_day(self, date: datetime) -> datetime:
        """Get the next working day from a given date"""
        next_date = date + timedelta(days=1)
        while not self.is_working_day(next_date):
            next_date += timedelta(days=1)
        return next_date
    
    def initialize_simulation(self, num_workers: int = 50, 
                            initial_wage: float = 100.0,
                            target_wage: float = 120.0,
                            initial_employer_balance: float = 100000.0,
                            initial_strike_fund: float = 50000.0):
        """Initialize the simulation with workers, employer, and union"""
        
        # Use settings if provided, otherwise use defaults
        num_workers = self.settings.get('num_workers', num_workers)
        initial_wage = self.settings.get('initial_wage', initial_wage)
        target_wage = self.settings.get('target_wage', target_wage)
        initial_employer_balance = self.settings.get('initial_employer_balance', initial_employer_balance)
        initial_strike_fund = self.settings.get('initial_strike_fund', initial_strike_fund)
        initial_morale_range = self.settings.get('initial_morale_range', (0.5, 0.8))
        initial_savings_range = self.settings.get('initial_savings_range', (0.0, 1000.0))
        initial_strike_participation_rate = self.settings.get('initial_strike_participation_rate', 0.6)
        
        # Create workers
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
        
        # Generate networks
        self.employer_network = self.generate_employer_network()
        self.union_network = self.generate_union_network()
        
        # Set initial strike state
        for worker in self.workers:
            if random.random() < initial_strike_participation_rate:
                worker.state = 'striking'
            else:
                worker.state = 'not_striking'

    def calculate_private_morale(self, worker: Worker, morale_spec: str = 'sigmoid') -> float:
        """Calculate worker's private morale based on financial position and target wage"""
        wage_gap = (worker.target_wage - worker.current_wage) / worker.current_wage
        savings_change = (worker.net_earnings - worker.savings) / max(worker.savings, 1.0)
        
        if morale_spec == 'sigmoid':
            # Sigmoid specification from the paper - fixed implementation
            alpha, beta, gamma = 1.0, 1.0, 0.5  # These could be made configurable
            
            # Wage factor: higher when current wage is closer to target wage
            wage_factor = alpha / (1 + np.exp(-wage_gap * 10))  # Positive wage gap increases morale
            
            # Savings factor: higher when net earnings are positive relative to initial savings
            savings_factor = beta / (1 + np.exp(-savings_change * 5))  # Positive savings change increases morale
            
            # Previous morale factor
            previous_morale = gamma * worker.morale
            
            # Combine factors - ensure result is between 0 and 1
            combined_morale = wage_factor * savings_factor * previous_morale
            return max(0.0, min(1.0, combined_morale))
            
        elif morale_spec == 'linear':
            # Linear specification
            alpha, beta, gamma = 0.3, 0.3, 0.4
            linear_morale = alpha * wage_gap + beta * savings_change + gamma * worker.morale
            return max(0.0, min(1.0, linear_morale))
            
        else:  # 'no_motivation'
            # No motivation specification
            alpha, beta, gamma = 0.5, 0.3, 0.2
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
        alpha = self.settings.get('private_morale_alpha', 0.7)
        beta = self.settings.get('social_morale_beta', 0.3)
        
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
        
        # Pay wages to working workers
        daily_wage_cost = 0.0
        for worker in working_workers:
            daily_wage = worker.current_wage / 20  # Assuming 20 working days per month
            worker.receive_wage(daily_wage)
            daily_wage_cost += daily_wage
        
        # Distribute strike pay to striking workers
        self.union.distribute_strike_pay(striking_workers)
        
        # Calculate employer revenue
        self.employer.calculate_daily_revenue(len(working_workers), daily_wage_cost)
        
        # Collect union dues (monthly, but simplified here)
        if self.day_count % 20 == 0:  # Monthly dues collection
            self.union.collect_dues(self.workers)
    
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
        # Check if strike should end
        striking_count = sum(1 for w in self.workers if w.state == 'striking')
        total_members = sum(1 for w in self.workers if w.state in ['striking', 'not_striking'])
        
        # Strike ends if participation collapses
        if striking_count == 0:
            return 'strike_collapsed'
        
        # Strike ends if employer grants concession
        if self.employer.should_grant_concession():
            concession_amount = 5.0  # Could be made configurable
            for worker in self.workers:
                worker.current_wage += concession_amount
            self.employer.grant_concession(concession_amount * len(self.workers))
            return 'employer_conceded'
        
        # Union policy adjustments based on participation
        participation_rate = striking_count / total_members if total_members > 0 else 0
        
        # Adjust strike pay based on participation
        if participation_rate < 0.3 and self.union.strike_fund_balance > 10000:
            self.union.strike_pay_rate = min(0.8, self.union.strike_pay_rate + 0.1)
        elif participation_rate > 0.8:
            self.union.strike_pay_rate = max(0.3, self.union.strike_pay_rate - 0.05)
        
        return 'continue'
    
    def run_daily_cycle(self):
        """Run one day of the simulation"""
        if not self.is_working_day(self.current_date):
            self.current_date += timedelta(days=1)
            return 'weekend'
        
        # Update worker morale
        for worker in self.workers:
            self.update_worker_morale(worker)
        
        # Process participation decisions
        self.process_participation_decisions()
        
        # Process financial flows
        self.process_daily_financial_flows()
        
        # Record simulation data
        striking_count = sum(1 for w in self.workers if w.state == 'striking')
        working_count = sum(1 for w in self.workers if w.state == 'not_striking')
        avg_morale = np.mean([w.morale for w in self.workers])
        
        self.simulation_data['dates'].append(self.current_date)
        self.simulation_data['striking_workers'].append(striking_count)
        self.simulation_data['working_workers'].append(working_count)
        self.simulation_data['employer_balance'].append(self.employer.balance)
        self.simulation_data['union_balance'].append(self.union.strike_fund_balance)
        self.simulation_data['average_morale'].append(avg_morale)
        
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
            'outcome': [outcome],
            'total_concessions': [self.employer.concessions_granted],
            'total_strike_pay': [self.union.strike_pay_distributed],
            'total_dues_collected': [self.union.dues_collected]
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Worker participation over time
        axes[0, 0].plot(self.simulation_data['striking_workers'], label='Striking', color='red')
        axes[0, 0].plot(self.simulation_data['working_workers'], label='Working', color='blue')
        axes[0, 0].set_title('Worker Participation Over Time')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Number of Workers')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average morale over time
        axes[0, 1].plot(self.simulation_data['average_morale'], color='purple')
        axes[0, 1].set_title('Average Worker Morale Over Time')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Average Morale')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Employer balance over time
        axes[1, 0].plot(self.simulation_data['employer_balance'], color='green')
        axes[1, 0].set_title('Employer Balance Over Time')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Balance ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Union balance over time
        axes[1, 1].plot(self.simulation_data['union_balance'], color='orange')
        axes[1, 1].set_title('Union Strike Fund Balance Over Time')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Balance ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
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
            'total_concessions': self.employer.concessions_granted,
            'total_strike_pay': self.union.strike_pay_distributed,
            'total_dues_collected': self.union.dues_collected
        }
        
        # Determine outcome
        if analysis['final_striking_workers'] == 0:
            analysis['outcome'] = 'strike_collapsed'
        elif analysis['total_concessions'] > 0:
            analysis['outcome'] = 'employer_conceded'
        else:
            analysis['outcome'] = 'ongoing'
        
        return analysis
