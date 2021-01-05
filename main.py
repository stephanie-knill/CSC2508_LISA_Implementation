"""
Created on Tues Dec 29 2020

@author: stephanieKnill
"""

import numpy as np
import math
import pandas
import scipy
from scipy import linalg

# Modules
import parameters
import visualizations
import utils


class LISA():
    """
    Defines a Learned Index Structure for Spatial Data (LISA)
    
    Args:
        initial_dataset:    Keys in the initial dataset
        page_size:          Predefined page size
        
    Attributes:
        border_points:      Pandas dataframe of border_points
        grid_cells:         List of grid cells in form [C_0, C_1, C_2, ...].
                            A grid cell is define by its lower and upper bounds for each axis. 
                            i.e. C = [l_0, u_0) x [l_1, u_1) x ... x [l_{d-1}, u_{d-1})
        learned_params:     List of parameters for each shard. Each shard has
                            a dictinoary with keys {"alpha", "beta"}
        local_models:       List of Local_Model objects that manage shards
        mapped_values:      Pandas dataframe of the mapped_values for each 
                            input key (maintains indices). Ascending order by mapped value
        pages:              List of pages that keys are stored in. Each page 
                            contains a list of key points.
        partition_mapping:  Dictionary of 'm_p' (list of border points for partitions) and
                            'partitions' (list of actual mapped partitions)
        shards:             Dictionary of shards called "shard_i". Each shard is
                            a list of tuples of (key_index, key_mapped_value) 
                            that are stored in the shard
        
    Callable Components: M, SP, L
        mapping_function(x):  Mapping function that takes a given key x in R^d
                              and assigns it a mapped value
        shard_pred_func(x):   Shard Prediction function that assigns a shard_id 
                              to a given key x \in R^d
        local_models:         List of Local_Model objects that manage shards  
    """
    
    
    def __init__(self, initial_dataset):     
        # Create empty dataframe of border_points
        index = np.arange(0, parameters.OPTIONS['num_partitions'][0] + 1)           #TODO: for now simplify 10 dataset 2 dimensions, 2) num partitions T_i = T_j
        self.border_points = pandas.DataFrame(np.nan, index=index, 
                                              columns=parameters.OPTIONS['datasets']['labels'])
        # Create empty list of grid cells
        self.grid_cells = []
      
        # Gogogogo!
        self.build_LISA(initial_dataset)
        
    def build_LISA(self, initial_dataset):
        """ Build LISA  """

        """ 1) Generation of Grid Cells """
        # Generate border points
        self._generate_border_points(initial_dataset)
        
        # Generate all grid cells
        self._generate_grid_cells()
        
        
        """ 2) Partially Monotonic Mapping Function M """
        # Generate mapped values for all keys in initial_dataset
        self._create_mapped_values(initial_dataset)
        
        
        """ 3) Monotonic Shard Prediction Function SP """
        # Train shard prediction function + partition into shards
        self._train_shard_prediction_func()
        
        """ 4) Construction of Local Models L """
        self._build_local_models(initial_dataset)
                
        print("LISA built")
    
    """ ======================== Local Model class ======================== """
    class Local_Model():
        """
        Define a local model L_i to model a shard S_i.
        
                
        Args/Attributes:
            Name:   Name of local model. Typically "local_model_i"
            PA:     List  of the addresses of the pages that overlap with S_i
            PM:     Sequence of sorted mapped values to partition the keys in S_i
        
        """
        def __init__(self, name, PA, PM):
            self.name = name
            self.PA = PA
            self.PM = PM
            
        # Search Functions: search within this local model
        def lbound(self, m):
            """
            Search in local model for mapped value m using lbound
            
            L_i.lbound(m) = j,   where L_i.PM[j-1] < m <= L_i.PM[j]
            """
            # Case 1: PM empty
            if self.PM == []:
                return 0
            
            # Case 2: m > m_max
            J = len(self.PM)
            m_max = self.PM[J-1]
            if m > m_max:
                return J-1
            
            # Case 3: m is somewhere in PM (perform binary search)
            j = self._binary_search_recursive_lbound(array=self.PM, element=m, 
                                                     start=0, end=len(self.PM))
            return j
            
        def _binary_search_recursive_lbound(self, array, element, start, end):
            if start > end:
                return -1
        
            mid = (start + end) // 2
            if element == array[mid]:
                return mid #lbound
            
            if (element <= array[mid]) and (element > array[mid-1]):
                return mid
        
            if element < array[mid]:
                return self._binary_search_recursive_lbound(array, element, start, mid-1)
            else:
                return self._binary_search_recursive_lbound(array, element, mid+1, end)
        
        
        def ubound(self,m):
            """
            Search in local model for mapped value m using ubound
            
            L_i.lbound(m) = j,   where L_i.PM[j-1] <= m < L_i.PM[j]
            """
            # Case 1: PM empty
            if self.PM == []:
                return 0
            
            # Case 2: m >= m_max
            J = len(self.PM)
            m_max = self.PM[J-1]
            if m >= m_max:
                return J-1
            
            # Case 3: m is somewhere in PM (perform binary search)
            j = self._binary_search_recursive_ubound(array=self.PM, element=m, 
                                                     start=0, end=len(self.PM))
            return j
        
        def _binary_search_recursive_ubound(self, array, element, start, end):
            if start > end:
                return -1
        
            mid = (start + end) // 2
            if element == array[mid]:
                return mid + 1 #ubound
            
            if (element < array[mid]) and (element >= array[mid-1]):
                return mid
        
            if element < array[mid]:
                return self._binary_search_recursive_ubound(array, element, start, mid-1)
            else:
                return self._binary_search_recursive_ubound(array, element, mid+1, end)
    
    
    def _build_local_models(self, inital_dataset):
        """
        Build local models L_i to manage each shard S_i.
        
        A shard S_i consists of Gamma keys.
        Each page can hold up to Omega keys.
        """
        Omega = parameters.OPTIONS['max_num_keys_per_page']
        
        # Create empty list of L_i's
        self.local_models = []
        
        # TODO: how do you create a page on disk?
        self.pages = []
        
        # Build each L_i
        for shard_name, shard_value in self.shards.items():
            # Get I_i = (k_0, ... , k_{gamma-1)), the keys in I that fal in S_i
            # Get m_i, the mapped values of I_i
            I_i = []
            m_i = []
            for point in shard_value:
                x_key = initial_dataset.iloc[point[0]]
                I_i.append(x_key)  # TODO: is this the index in the original dataset? or just the index for them
                m_i.append(point[1])
            
            # Instantiate PA, PM
            PA = []
            PM = []
            
            Gamma = len(m_i)
            # Case 1: all keys fit in one shard
            if Gamma <= Omega:
                # Allocate a new page
                # Store I_i in P
                address = len(self.pages)
                P = I_i
                self.pages.append(P)
                
                # Append address of P to L_i.PA
                PA.append(address)
            
            # Case 2: need to split keys into multiple pages
            else:
                # Partition I_i along their mapped values into several pieces
                # ensuring each piece contain less than Omega
                W = int(np.ceil(Gamma / Omega))   # Number of pages need
                delta = int(np.ceil(Gamma / W))   # Number of keys will put per page (ceiling)
                
                for j in range(1, W+1):
                    # TODO: Allocate a new page on disk
                    P = None
                    ind_lower = (j-1) * delta
                    if j == W:
                        ind_upper = -1
                        
                        # Append last point?
                        PM.append(m_i[ind_upper])
                    else:
                        ind_upper = j * delta-1
                                            
                        # Append m_{i * delta} to L_i.PM
                        m = m_i[ind_upper]
                        PM.append(m)
                                                
                        # Append first point?
                        if j == 1:
                            PM.append(m_i[ind_lower])
                    
                    # Store keys in each partition to a different page
                    P = I_i[ind_lower : ind_upper]
                                    
                    # Append addresses of P's to L_i.PA
                    address = len(self.pages)
                    PA.append(address)
                    self.pages.append(P)
                    
            # Save as a Local_Model object
            local_model_name = "local_model_%s" %shard_name.split('_')[1]
            local_model = self.Local_Model(local_model_name, PA, PM)
            
            self.local_models.append(local_model)
            
    
    """ ======================== LISA Functions ======================== """
    def range_query(self, qr):
        """
        Perform a range query on the index for a given query rectangle qr
            qr = [l_0, u_0) x ... x [l_{d-1}, u_{d-1})
        
        to obtain the keys R that fall in qr.                             
        
        Input:
            qr:     Query rectangle

        Returns:
            R:      A list of keys falling in qr                            
        """
        
        # 1) Get cells that overlap qr
        cells = self._get_cells_overlap_qr(qr)
        
        # 2) Decompose qr into a union of smaller qr's that intersect only 1 cell
        cells_smaller = self._decompose_cells_by_qr(cells, qr)
        
        # 3) TODO: Merge consecutive small rectangles (do if have time)
        
        # Set PageAddrs to be an empty set
        PageAddrs = set()
        
        # Iterate across all cells that cover qr
        for j, cell in enumerate(cells_smaller):
            # 4) Calculate mapped values of all qr's vertices
            x_l = [cell[0][0], cell[1][0]]
            x_u = [cell[0][1], cell[1][1]]
            
            x_l_map = self.mapping_function(pandas.DataFrame([x_l], columns=parameters.OPTIONS['datasets']['labels']).iloc[0])
            x_u_map = self.mapping_function(pandas.DataFrame([x_u], columns=parameters.OPTIONS['datasets']['labels']).iloc[0])
           
            # 5) Calculate SP for all rectangles
            # Handle edge cases: mapped values larger or smaller than mapped partitions            
            try:
                k_l = int(np.floor(self._shard_pred_func_xmap(x_l_map)))
                if k_l < 0:
                    k_l = 0
                if k_l >=len(self.shards):
                    k_l = len(self.shards) -1
            except:
                k_l = 0                
            try:
                k_u = int(np.floor(self._shard_pred_func_xmap(x_u_map)))
                if k_u < 0:
                    k_u = 0
                if k_u >= len(self.shards):
                    k_u = len(self.shards) -1
            except:
                k_u = len(self.shards) -1
        
            # 6) Local Models: select pages that overlap with qr
            # Binary search: lbound
            l = self.local_models[k_l].lbound(x_l_map)
            
            # if l == -1:
            # Iterate across all page addresses in L_i.PA
            for s in range(l, len(self.local_models[k_l].PA) - 1):
                try:
                    PageAddrs.add(self.local_models[k_l].PA[s])
                except:
                    continue
            
            for t in range(k_l+1, k_u-1):
                for elem in self.local_models[t].PA:
                    PageAddrs.add(elem)
        
            # Binary search: ubound
            u = self.local_models[k_u].lbound(x_u_map)
            
            
            # if u != -1:
            for s in range(0, u):
                try:
                    PageAddrs.add(self.local_models[k_u].PA[s])
                except:
                    continue
            
        # Return all the keys contained in R
        R = []
        for P in PageAddrs:
            # Get page
            page = self.pages[P]
            
            # Each page contains a list of keys
            # Check if each key is in qr
            for key in page:
                if (self._check_key_in_qr(key, qr)):
                    # If key in qr, add to R
                    R.append(key)
        
        # Not necessary...but make R an easy to read dataframe
        index = np.arange(0, len(R))
        R_df = pandas.DataFrame(np.nan, index=index, columns=parameters.OPTIONS['datasets']['labels'])
        for ind, val in enumerate(R):
            R_df.iloc[ind] = val
        
        # Sort keys
        R_df = R_df.sort_values(by=parameters.OPTIONS['datasets']['labels'][0])
        
        return R_df
            
    
    def _check_key_in_qr(self, key, qr):
        """
        Check if a given key (panda Series) is contained within a query rectangle qr.
            qr = [l_0, u_0) x ... x [l_{d-1}, u_{d-1})
        
        Returns:
            boolean:   True or False
        """
        for ind, axis in enumerate(qr):
            if (axis[0] <= key[ind]) and (key[ind] <= axis[1]):
                continue
            else:
                return False
          
        return True
    
    
    def _decompose_cells_by_qr(self, cells, qr):
        """
        Decompose qr into a union of smaller qr's that intersect only 1 cell
        
        Input:
            cells:              List of cells that cover qr
            qr:                 Query rectangle
        Returns
            cells_smaller:      List of cells that are intersection of qr and cells
        """
        cells_smaller = []
        for cell in cells:
            cell_smaller = []
            for dim in range(len(cell)):
                if qr[dim][0] < cell[dim][0]:
                    lb = cell[dim][0]
                else:
                    lb = qr[dim][0]
                
                if qr[dim][1] > cell[dim][1]:
                    ub = cell[dim][1]
                else:
                    ub = qr[dim][1]
                
                cell_smaller.append([lb, ub])
            cells_smaller.append(cell_smaller)
        
        return cells_smaller
        
    
    def _get_cells_overlap_qr(self, qr):
        """
        Find cells that contain a given query rectangle qr
            qr = [l_0, u_0) x ... x [l_{d-1}, u_{d-1})
        
        A grid cell is define by its lower and upper bounds for each axis. 
            C = [l_0, u_0) x [l_1, u_1) x ... x [l_{d-1}, u_{d-1})                           
        
        Input:
            qr:     Query rectangle

        Returns:
            cells:   List of grid cells that overlap qr.                     
        """
        #TOOD: if have time, make this more abstract (i.e. do not hardcode lat, lon)
        # Get cells that contain corners of qr
        qr_ll = [qr[0][0], qr[1][0]]
        qr_ul = [qr[0][1], qr[1][0]]
        qr_lu = [qr[0][0], qr[1][1]]
        qr_uu = [qr[0][1], qr[1][1]]
        
        # Convert to dataframe
        df_ll = pandas.DataFrame([qr_ll], columns=parameters.OPTIONS['datasets']['labels']).iloc[0]
        df_ul = pandas.DataFrame([qr_ul], columns=parameters.OPTIONS['datasets']['labels']).iloc[0]
        df_lu = pandas.DataFrame([qr_lu], columns=parameters.OPTIONS['datasets']['labels']).iloc[0]
        df_uu = pandas.DataFrame([qr_uu], columns=parameters.OPTIONS['datasets']['labels']).iloc[0]
        
        # Get cell numbers
        cell_ll = self._get_cell_num_contains_x(df_ll)
        cell_ul = self._get_cell_num_contains_x(df_ul)
        cell_lu = self._get_cell_num_contains_x(df_lu)
        cell_uu = self._get_cell_num_contains_x(df_uu)
        
        cell_nums = {cell_ll, cell_ul, cell_lu, cell_uu}
        
        # Get rest of cell numbers
        diff_lon =  cell_ul - cell_ll
        diff_lat =  cell_uu - cell_ul #Very hacky but out of time; fix when make more abstract
        
        
        # Iterate across lon
        def get_cells_across_lon(curr_cell, diff, cell_nums):
            cells_more = diff // parameters.OPTIONS['num_partitions'][0]
            while cells_more > 0:
                cell_num = curr_cell + parameters.OPTIONS['num_partitions'][0] * cells_more
                cell_nums.add(cell_num)
                cells_more = cells_more - 1
            return cell_nums
       
        curr_cell = cell_ll
        for i in range(diff_lat+1):
            cell_nums.add(curr_cell)
            cell_nums = get_cells_across_lon(curr_cell, diff_lon, cell_nums)
            curr_cell = curr_cell + 1
        
        # Order cell numbers into a list
        cell_nums = list(cell_nums)
        
        # Return as cells (not cell numbers)
        cells = []
        for cell_num in cell_nums:
            try:
                cell = self._get_grid_cell(cell_num)
                cells.append(cell)
            except:
                pass
        
        return cells
    
    
    def shard_pred_func(self, x):
        """
        After the Shard Prediction function SP is trained, assign a shard id to a 
        new point x \in R^d.
            shard_id = floor(SP(m)),   where m is the mapped value of x
        
        The Shard Prediction Function SP : R --> [0, +inf) is given by
            SP(x) = F_i(x) + i*D,   for x in R
                                    where i is binary search in M_p for x
                                    D= ceil((V+1) / Psi) is number of shards for F_I
        Input:
            x:            Datapoint x \in R^d
            
        Output:
            shard_id:     Unique shard_id  of the shard that the key x should be stored in 
        """
        # Compute mapped value
        x_map = self.mapping_function(x)
        
        return self._shard_pred_func_xmap(x_map)
    
    
    def _shard_pred_func_xmap(self, x_map):
        # Get appropriate alpha, beta and i for f_i function        
        i = self._get_map_partition_contains_x(x_map)
        alpha = self.learned_params[i]['alpha']
        beta = self.learned_params[i]['beta']
        V_plus_1 = self.learned_params[i]['V_plus_1']

        return self._assign_shard_id(x_map, alpha, beta, i, V_plus_1)
    
    
    def _assign_shard_id(self, x, alpha, beta, i, V_plus_1):
        """
        Compute shard ID for a given mapped value x in R.

        Input:
            x :         Mapped x value
            alpha :     Learned alpha vector for F_i
            beta :      Learned beta vector for F_i
            i :         Mapped partition number, i.e. F_i has domain [m_{i-i}~, m_i~)]

        Returns:
            shard_id:   Shard id assigned to given mapped x value

        """
        # Compute F_i(x) = f_i(m) / Psi
        Psi = parameters.OPTIONS['est_avg_num_keys_shard']
        f_i = self._piecewise_f_function(alpha, beta, x)
        F_i = f_i / Psi
        
        # Compute D
        D = np.ceil((V_plus_1) / Psi)
    
        sp = F_i + i*D
        shard_id = np.floor(sp)
        
        if shard_id < 0:
            shard_id = 0 #TODO: why is f_i sometimes very small negative (henec shard_id when i=0 is negative?)
            # raise ValueError("Shard id must be non-negative (instead, have %s)" %shard_id)
        
        return shard_id
    
    def _get_map_partition_contains_x(self, x):
        """
        Return index for mapped partition that contains x
        """
        # TODO: how do you classify x points that are not in these partitions
        
        m_p = self.partition_mapping['m_p']
        
        # Handle edge case: x = m_U~ (i.e. x is the upper border)
        if x == m_p[-1]:
            return len(m_p) - 2
        
        # Binary search for partition number      
        ind = self._binary_search_recursive(m_p, x, 0, len(m_p))
        return ind
    
    def _binary_search_recursive(self, array, element, start, end):
        if start > end:
            return -1
    
        mid = (start + end) // 2
        if element == array[mid]:
            return mid
        
        if (element >= array[mid]) and (element < array[mid+1]):
            return mid
    
        if element < array[mid]:
            return self._binary_search_recursive(array, element, start, mid-1)
        else:
            return self._binary_search_recursive(array, element, mid+1, end)
        
    
    def _train_shard_prediction_func(self):
        """
        1) Train Shard Prediction function SP, specifically F_i by obtaining 
        optimal alpha and beta parameters. This is saved in self.learned_params
        
        The shard prediction functin for x \in R
        SP(x) = F_i(x) + i*D
        
        where F_i: monotonic regression model with domain [m_{i-1}*, m_i*]
              D:   number of shards that F_I generates
                    -> D = ceil[(V+1) / Psi)]
              i:   binarry search (M_p, x)
              V+1: number of mapped values that each F_i needs to process
              Psi: estimated average number of keys falling in a shard
       
        2) Then, partition the points into shards
        
        """
        # Generate number list M_p of that partitions the mapped values
        self._partition_mapped_values(parameters.OPTIONS['num_partitions_mapped_values'])
        m_p = self.partition_mapping['m_p']
        partitions = self.partition_mapping['partitions']
        
        # Save alpha, beta for training
        self.learned_params = []
        
        # Save shard partitions
        self.shards = {} #TODO: initialzie it to correct number of shards?
        
        # For each mapped value partition
        for i in range (0, len(partitions)):
            # Get x_map
            x_map = partitions[i]           
            V_plus_1 = len(x_map)
            ind = np.arange(0, V_plus_1)
            
            # Train F_i
            print("\n============== Training for partition %s ==============" %i)
            alpha, beta = self._train_function_f(x_map['mappped_value'].values.tolist(), ind)
            self.learned_params.append({'alpha': alpha, 'beta' : beta, 'V_plus_1' : V_plus_1})
            
            # Use F_I to partition all the mapped points into shards
            for j in range(len(x_map)): #x is a mapped point
                x_val = x_map.iloc[j]['mappped_value']
                shard_id = int(self._assign_shard_id(x_val, alpha, beta, i, V_plus_1))
                                             
                # print(shard_id)
                # Add mapped point to appropriate shard (do I need to add its original value too??)
                col_name = "shard_%s" %shard_id
                
                # Save as tuple (key_index, mapped_value), where key_index is the key's index in the initial dataset
                if col_name in self.shards:
                    self.shards[col_name].append((x_map.index[j], x_val))
                else:
                    self.shards[col_name] = [(x_map.index[j], x_val)]
            
            
    def _train_function_f(self, x_map, y):
        """
        Monotonic piecewise linear function f_i(x_map).
        
        Need to find the optimal values for beta (breakpoints) and alpha
            alpha = (a~, a_0, a_1, ... , a_sigma)
            beta =  (    b_0, b_1, ... , b_sigma)
        where sigma+1 is the number of breakpoints.
        
        To solve, we will first fix beta and solve for alpha (A alpha = y);
        since A^T A is symeetric, can solve for alpha via cholesky
        Then we will search for the optimal beta.
        
        
        
        Args:
            x_map:      Sorted mapped values (m_0, .., m_V), where V+1 is the 
                        number of mapped values that F_i processes
            y:          Indices of x_map (0, ... , V)
        
        Returns:
            alpha
            beta
        """
        # Get parameters
        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']
        V_plus_1 = len(x_map)
        psi = parameters.OPTIONS['est_avg_num_keys_shard']
        
        # Initialize beta
        beta = np.empty(sigma_plus_1)
        beta[:] = np.NaN
        
        beta[0] = x_map[0]
        for i in range(1, sigma_plus_1):
            ind = int(np.floor(i * (V_plus_1 - 1) / psi))
            beta[i] = x_map[ind]
            
        # Find optimal Alpha for given breakpoint Beta
        A  = self._form_A_matrix(x_map, y, beta)
        alpha = self._cholesky_factorization(A, y)        
        
        # Train        
        num_iterations = 1
        while True:            
            # Perform cell search along direction s to find learning rate  lr^(k) >=0
            lr_new, loss_new, alpha_new, beta_new = self._compute_learning_rate(alpha, beta, x_map, y)
              
            # if num_iterations % 50 == 0:
            #     print ("Iteration: %d - Error: %.4f" %(num_iterations, loss))
            print("Iteration %s, Loss is %s" %(num_iterations, loss_new))            
            
            # Stopping Conditions
            if (num_iterations == parameters.OPTIONS['max_iterations']):
                print("Max iteration of %s reach. Terminating" %num_iterations)
                break
                
            if np.abs(sum(beta-beta_new)) < parameters.OPTIONS['tolerance']:
                print("Converged")
                break
            
            if lr_new == 0:
                print("Failed to find new learning rate. Terminating.")
                break
        
            num_iterations += 1
            alpha = alpha_new
            beta = beta_new
        
        # Return the optimal alpha and beta
        return alpha, beta
    
    def _compute_learning_rate(self, alpha, beta, x_map, y):
        """
        Find a new learning rate lr>=0 that minmizes loss function, while still 
        satisfying sum a_i's non-negativity.
        
        Perform a grid search over all possible hyperparameters
        
        Returns:
            learning_rate:      Returns new learning rate to use for k-th iteration
        """
        
        # Define the grid search parameters
        if  beta[0] == x_map[0]: #TODO: remove this once testing done (only use lr=1 for first iteration)
            learn_rates = [1]
            print("First iteration for this partition")
        else:
            learn_rates = parameters.OPTIONS['learning_rates']
        
        # TODO: remove after test
                
        # Initialization for search
        loss_best = self._square_loss_function(alpha, beta, x_map, y)
        lr_best = 0  # Placeholder in case all learning_rates fail
        beta_best = beta
        alpha_best = alpha
        
        # Compute step direction
        A  = self._form_A_matrix(x_map, y, beta)
        s = self._compute_s(alpha, A, beta, x_map, y)
        
        for lr in learn_rates:
            # See if learning_rate valid
            try:
                # Update beta: B^(k+1) = B^(k) + lr^(k)*s^(k)
                # Ignore first entry of s vector (beta and alpha dimension are different)
                beta_new = beta + lr * s[1:]  
                valid1 = self._check_sum_a_non_negative(beta_new, x_map, y)
                valid2 = self._check_beta_increasing(beta_new)
                if not (valid1 and valid2):
                    continue
            except:
                # Go back to beginning of for loop
                print("Invalid learning rate")
                continue
            
            # Find optimal Alpha for this particular breakpoint Beta_new
            A  = self._form_A_matrix(x_map, y, beta_new)
            alpha_new = self._cholesky_factorization(A, y)
            
            loss = self._square_loss_function(alpha_new, beta_new, x_map, y)
            
            if (loss < loss_best):
                loss_best = loss
                lr_best = lr
                beta_best = beta_new
                alpha_best = alpha_new
         
        return lr_best, loss_best, alpha_best, beta_best
                  
            
    def _compute_s(self, alpha, A, beta, x_map, y):
        # Get parameters
        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']
        V_plus_1 = len(x_map)
            
        # Initialization: compute input matrices based on alpha & beta 
        K = np.diag(alpha)
        
        G = np.empty((sigma_plus_1+1, V_plus_1))
        G[:] = np.NaN
        G[0,:] = -1
        for row in range(1, sigma_plus_1+1):
            b = beta[row - 1]
            for col in range(V_plus_1):
                x_i = x_map[col]
                if x_i >= b:
                    val = -1
                else:
                    val = 0
                G[row, col] = val

        # Form Y matrix which is positive definite (symmetric, positive pivots)
        Y = 2 * K.dot(G).dot(np.transpose(G)).dot(np.transpose(K))
        
        # Form g: g = 2*KGr, where r = A alpha - y
        r = A.dot(alpha) - y
        g = 2 * K.dot(G).dot(r)
        
        # Update in direction s= -Y^{-1} g
        try:
            s = -1 * (np.linalg.inv(Y)).dot(g)
        except:
            print(beta)  #TODO: fix why Y is sometimes singular????
            print(-1 * np.linalg.inv(Y))

        return s   
            
    
    def _check_beta_increasing(self, beta):
        """
        Check if a given beta vector is valid:
            b_0 < b_1 < ... < b__sigma
        """
        prev = beta[0]
        for i in range(1, len(beta)):
            if prev > beta[i]:
                return False
            prev = beta[i]
            
        return True
            
    
    def _check_sum_a_non_negative(self, beta, x_map, y):
        """ 
        Check if a given learning_rate lr produces a beta that gives a valid alpha.
        Must satisfy
            sum_{i=0)^n a_i >= 0,    for all 0 <= n <= sigma
        
        Returns
            True or False:     Whether the sum of a_i's is non-negative
        """
        
        # Compute alpha
        A  = self._form_A_matrix(x_map, y, beta)
        
        try:
            alpha = self._cholesky_factorization(A, y)
        except:
            return False
        
        
        # Verify alpha valid
        for i in range(len(alpha)):
            sum_a = np.sum(alpha[:i])
            
            if sum_a < 0:
                return False
        
        return True
        
        
    def _square_loss_function(self, alpha, beta, x, y):
        """
        Compute square loss function
        L(alpha, beta) = \sum_{i=0}^V (f(x_i) - y_i)^2
        
        Input:
            alpha:      Vector alpha = (a~, a_0, a_1, ... , a_sigma)
            beta:       Vector beta =  (    b_0, b_1, ... , b_sigma)
            x:          Vector of mapped x values
            y:          Vector of indices of x
        
        Returns:
            loss:       Square loss function
        """        
        
        # Iterate over all V+1 points
        loss = 0
        for i in range(len(x)):
            f_i = self._piecewise_f_function(alpha, beta, x[i])
            l = (f_i - y[i])**2
            loss += l
        
        return loss
        
    
    def _piecewise_f_function(self, alpha, beta, x_map):
        """ 
        Compute piecewise function f(x_map)
        Input:
            alpha:      Vector alpha = (a~, a_0, a_1, ... , a_sigma)
            beta:       Vector beta =  (    b_0, b_1, ... , b_sigma)
            x_map:      Mapped value of x
        Returns:
            f:          f(x_i)
        """
        f = 0
        if beta[0] <= x_map:
            f += alpha[0] + alpha[1]*(x_map - beta[0])
        for i in range(1, len(beta)):
            if x_map >= beta[i]:
                f += alpha[i+1]*(x_map - beta[i])
            else:
                break
            
        return f
    
    
    def _form_A_matrix(self, x_map, y, beta):
        """Given a fixd beta value, solve for alpha 
        
        Returns:
            A:      Matrix for least squares equation A alpha = y
        """
        # Get parameters
        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']
        V_plus_1 = len(x_map)
        
        # Generate A: (V+1) x (sigma+2) for row x col
        A = np.empty((V_plus_1, sigma_plus_1+1))
        A[:] = np.NaN
        A[:, 0] = 1
        
        # Create Indicator function
        indicator = np.empty((V_plus_1, sigma_plus_1))
        indicator[:] = np.NaN
        indicator[:, 0] = 1
        
        # Iterate across rows then cols
        for row in range(0, V_plus_1):
            x_i = x_map[row]
            for col in range(1, sigma_plus_1):
                b = beta[col]
                if x_i >= b:
                    indicator[row, col] = 1
                else:
                    indicator[row, col] = 0
        
        # Fill in A
        for row in range(0, V_plus_1):
            x_i = x_map[row]
            for col in range(1, sigma_plus_1+1):
                b = beta[col-1]                
                val = (x_i - b) * indicator[row, col-1]
                A[row, col] = val
        
        return A


    def _cholesky_factorization(self, A, y):
        """"
        For least squares equation A alpha = y, solve for alpha.
        Since A^T A is symeetric, can solve for alpha via cholesky factorization
        
        Cholesky Factorization Algorithm: http://www.math.iit.edu/~fass/477577_Chapter_5.pdf, page 50
        
        Input:
            A:      Input matrix 
            y:      Vector
        
        Returns:
            alpha:  Vector solution to least squares equation A alpha = y
        """
        # Compute A^T A matrix
        A_T = np.transpose(A)
        LHS = A_T.dot(A)
        
        # # Compute A^T y vector
        RHS = A_T.dot(y)
        
        # Compute choleksy decomposition: A^T A = R^T R        
        R = scipy.linalg.cho_factor(LHS)
        
        # Cholesky factorization
        alpha = scipy.linalg.cho_solve(R, RHS)
        
        return alpha
    
  
    def _partition_mapped_values(self, U):
        """
        Compute a list of numbers M_p = [m_1*, ... , m_U*] that evenly partitions
        the mapped values into U similar sized partitions, i.e. the number of 
        mapped values in [m_{i-1}*, m_i*] is almost the same for all 0 <= i <= U
        
        Input:
            U:          Number of partitions for the mapped values.
        
        Returns:
            m_p:        Number list that partitions the mapped values
        """
        
        # mapped_values =['mappped_value']
        partitions = np.array_split(self.mapped_values, U)
        
        # M_p: take first index of each partition, plus last index
        m_p = []
        for val in partitions:
            m_p.append(val['mappped_value'].iloc[0])
            
        m_p.append(partitions[-1].iloc[-1]['mappped_value'])  #TODO: need to make it a little bigger? otherwise not included in range [m_i*, m_{i+1}*)]
        
        self.partition_mapping = {"m_p" : m_p, "partitions" : partitions}

    
    def _create_mapped_values(self, dataset):
        """
        Compute the mapped values for all points in a dataset and save to a
        pandas dataframe self.mapped_values.
        
        Input:
            dataset:    Input dataset (pandas dataframe) to generate mapped values from
        """
        
        # Create empty dataframe of the mapped values
        index_mv = np.arange(0, len(dataset))
        self.mapped_values = pandas.DataFrame(np.nan, index=index_mv,
                                              columns=['mappped_value'])        
        
        # Compute + save mapped values
        for index, row in dataset.iterrows():
            map_value = self.mapping_function(row)
            self.mapped_values['mappped_value'][index] = map_value
        
        # Sort mapped values
        self.mapped_values = self.mapped_values.sort_values(by=['mappped_value'])
    
    
    def mapping_function(self, x):
        """
        Monotonic mapping function M(x) that takes an input data point x=(x_0, x_1)
        and maps it to 0 \cup R^{+}, i.e. to [0, +inf)
            M = i + (mu(H_i)) / (mu(C_i))
            
        where cell C_i is the cell that contains x s.t. 
                   C_i = [l_0, u_0] x [l_1, u_1] 
              cell H_i = [l_0, x_0] x [l_1, x_1], where l_0 and l_1 are the lower bounds of C_i
              mu() is the Lebesgue measure function.
        Input:
            x:      Input data point in form pandas dataframe????
            
        Returns
            M:      Mapping M(x)
        """
        # TODO: check each x_i >=0
        for x_i in x:
            assert x_i >= 0
        
        # Get cell C_i that contains x
        cell_num = self._get_cell_num_contains_x(x)
        cell = self.grid_cells[cell_num]
        
        # Create cell H_i
        H_cell = self._get_grid_cell(cell_num)   #Copy list problems, so for now just recompute
        for j in range(len(H_cell)):
            H_cell[j][1] = x[j]
            
        # Lebesgue Measure
        mu_h = self._lebesgue_measure(H_cell)
        mu_c = self._lebesgue_measure(cell)    

        # Mapping Function           
        M = cell_num  + mu_h / mu_c  
            
        return M
    
   
    def _lebesgue_measure(self, cell):
        """Compute lebesgue measure for a cell. Since we only have one point, x,
        in our subset, this is just the area/volume of the cell
        
        Returns:
            volume:      Lesbesgue measure of given cell
        """
        
        interval_lengths = []
        for interval in cell:
            length = interval[-1] - interval[0]
            interval_lengths.append(length)
            
        volume = np.prod(interval_lengths)
        return volume
        
        
    def _get_cell_num_contains_x(self, x):
        partition_indices = []
        
        # Check along each axis in border_points
        dim = len(parameters.OPTIONS['datasets']['labels'])
        for i in range(dim):
            col_name = parameters.OPTIONS['datasets']['labels'][i]
            axis = self.border_points[col_name]
            found = False
            
            for j in range(0, len(axis)-1):
                if x[col_name] <= axis[j+1]:
                    partition_indices.append(j)
                    found = True
                    break
            if found == False:
                raise ValueError("Input datapoint not within borderpoints along %s axis" %col_name)
       
        
        # Determine grid cell number based on partition indices
        # t = T_1*j_0 + j_1 for cell C_t
        cell_num = parameters.OPTIONS['num_partitions'][1] * partition_indices[0] + partition_indices[1]
        
        return cell_num
        
        
    def _generate_grid_cells(self):
        """"
        Generate all grid cells based on self.border_points and saves them in self.grid_cells.
        
        self.grid_cells in format [C_0, C_1, C_2, ...]          
        
        """
        # Number of cells = T_1 * ... * T_{d-1}
        num_cells = np.prod(parameters.OPTIONS['num_partitions'])
        
        for i in range(num_cells):
            grid_cell = self._get_grid_cell(i)
            self.grid_cells.append(grid_cell) 
        
    
    def _get_grid_cell(self, cell_num):
            """
            A grid cell is define by its lower and upper bounds for each axis. 
            i.e. C = [l_0, u_0) x [l_1, u_1) x ... x [l_{d-1}, u_{d-1})
                                                      
            Cell numbering is C_0, C_1, ... C_{T_1 x T_2 ... T_{d-1}} which is
            organized (for visualization, in 2D).
            
            For simplicity, we are only doing it in 2 dimensions, thus the formula is
                  t = T_1*j_0 + j_1 for cell C_t
            where t     : cell number
                  T_i   : number of partitions in axis i
                  j_i   : Upper bound/border for cell C_T along  axis i
            
            # Again, for simplicity, T_i = T_j
            
            ^ X_1
            |
            |
            C_2     C_5
            C_1     C_4
            C_0     C_3 ______> X_0
            
            Returns:
                grid_cell:      Grid cell in format [[l_0, u_0], [l_1, u_1]]
            
            
            """
            dim = len(parameters.OPTIONS['datasets']['labels'])
            T = parameters.OPTIONS['num_partitions']
            grid_cell = []
            j=[]
            
            # j_0 = cell_num // T_1
            j_0 = cell_num // T[1]
            j.append(j_0)
            # print("cell_num is: %s" %cell_num)
            # print("j_0 is: %s" %j_0)
            
            # j_1 = t - (T_1 * j_0)
            j_1 = cell_num - ( T[1] * j_0)
            j.append(j_1)
            # print("j_1 is: %s" %j_1)
            
            # Grab lower & upper bounds along each axis
            for i in range(dim):
                axis = parameters.OPTIONS['datasets']['labels'][i]
                # lb
                lb = self.border_points[axis][j[i]]
                
                
                # ub
                ub = self.border_points[axis][j[i]+1]
                
                # Add bounds to cell
                grid_cell.append([lb, ub])
            
            return grid_cell
                        
            
    def _generate_border_points(self, dataframe):
        """
        Partition the space into a series of grid cells based on the data 
        distribution along a sequence of axes and numbering the cells along these axes.
        
        Stores the border points generated by the partition operation in
        self.border_points.

        Args:
            dataframe:      Input dataset (pandas dataframe)

        """
        
        dim = len(parameters.OPTIONS['datasets']['labels'])
        
        
        # For each axis x_i
        for  i  in range(dim):
            # Partition into T_i parts; save only border_points
            col_name = parameters.OPTIONS['datasets']['labels'][i]
            self._partition_1D_dataframe(parameters.OPTIONS['num_partitions'][i],
                                        col_name,
                                        dataframe[col_name])
            

    def _partition_1D_dataframe(self, num_partitions, col_name, dataframe):
        """
        Given a 1D Pandas dataframe, partition into num_partitions of similar size.
        Save the border points in self.border_points for col_name

        Args:
            num_partitions:         Number of partitions
            col_name:               Name of column/axis to conduct partition on
            dataframe:              Input 1D pandas dataframe

        """
        # Sort points
        sort = dataframe.sort_values()
        
        # Partition into T_i parts
        chunk_size = math.ceil(len(sort) / num_partitions)  #TODO: fix if not perfect divide??
         
        # Lower Bound: origin
        self.border_points[col_name].iloc[0] = 0
        
        # Internal Points: take midpoint of extremes of adjacent paritions
        for i in range(1, num_partitions):
            lower = sort.iloc[i * chunk_size-1]
            
            # TODO: very hacky fix for not perfect divide (look up np.array_split)
            if i == num_partitions:
                upper = sort.iloc[-1]
            else:
                
                upper = sort.iloc[i * chunk_size]
            
            border = (lower + upper) / 2
            self.border_points[col_name].iloc[i] = border
        
        # Upper Bound: ceiling of last point
        ub = math.ceil(sort.iloc[-1])
        self.border_points[col_name].iloc[-1] = ub



''' ###################### Testing Range Queries ###################### '''
def generate_qr(self, LISA_index):
    """
    Randomly generate a query recetangle (qr)
    
    Input:
        LISA_index:   a LISA object
    
    Returns:
        qr:           A query rectangle
    """



if __name__ == '__main__':    
          
    ''' ###################### Construction of Index ###################### '''
    
    initial_dataset = utils.load_dataset()  
    index = LISA(initial_dataset)
                 
    ''' ###################### Query Processing ###################### '''

    # 1) Range Queries 
    qr = [[20, 28], [37, 40]]
    R = index.range_query(qr)
    
    # 2) KNN Queries 
    
    
    ''' ###################### Create Visualizations for Performance ###################### '''