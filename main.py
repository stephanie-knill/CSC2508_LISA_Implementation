"""Created on Tues Dec 29 2020@author: stephanieKnill"""#import osimport numpy as npimport mathimport pandasimport scipyfrom scipy import linalg# Modulesimport parametersimport query_processingimport visualizationsimport utilsclass LISA():    """    Defines a Learned Index Structure for Spatial Data (LISA)        Args:        initial_dataset:    Keys in the initial dataset        page_size:          Predefined page size            Attributes:        keys_init:          Keys in the initial dataset        page_size:          Predefined page size        map_func:           Partially monotonic mapping function M        shard_pred_func:    Shard prediction function         local_models:       Local models for all shards    """            def __init__(self, initial_dataset, page_size):        # self.initial_dataset = initial_dataset    #TODO: this might be too big to store as part of the index        self.page_size = page_size                # Create empty dataframe of border_points        index = np.arange(0, parameters.OPTIONS['num_partitions'][0] + 1)           #TODO: for now simplify 10 dataset 2 dimensions, 2) num partitions T_i = T_j        self.border_points = pandas.DataFrame(np.nan, index=index,                                               columns=parameters.OPTIONS['datasets']['labels'])        # Create empty list of grid cells        self.grid_cells = []              # Gogogogo!        self.build_LISA(initial_dataset)                    def build_LISA(self, initial_dataset):        """ Build LISA  """        """ 1) Generation of Grid Cells """        # Generate border points        self.generate_border_points(initial_dataset)                print(self.border_points) #TODO: remove once done testing                # Generate all grid cells        self.generate_grid_cells()                        """ 2) Partially Monotonic Mapping Function M """        # Generate mapped values for all keys in initial_dataset        self.create_mapped_values(initial_dataset)                        """ 3) Monotonic Shard Prediction Function SP """        self.shard_prediction_func()                """ 4) Construction of Local Models L """        return 0                def shard_prediction_func(self):        """        Train Shard Prediction function SP.                The shard prediction functin for x \in R        SP(x) = F_i(x) + i*D                where F_i: monotonic regression model with domain [m_{i-1}*, m_i*]              D:   number of shards that F_I generates                    -> D = ceil[(V+1) / Psi)]              i:   binarry search (M_p, x)              V+1: number of mapped values that each F_i needs to process              Psi: estimated average number of keys falling in a shard                               """        # Generate number list M_p of that partitions the mapped values        m_p, partitions = self.partition_mapped_values(parameters.OPTIONS['num_partitions_mapped_values'])                # For each mapped value partition        for i in range (0, len(m_p)-1):            # Get x_map            x_map = partitions[i]            V_plus_1 = len(x_map)            ind = np.arange(0, V_plus_1)                        # Train F_i            print("\n============== Training for partition %s ==============" %i)            alpha, beta = self.train_function_f(x_map, ind)                                    # TODO: save alpha, beta trainig. that way you can easily compute f_i at will            # f_i = self.piecewise_f_function(alpha, beta, x_map[i])                    return 0 #TODO: finish once fix helper function                                    def train_function_f(self, x_map, y):        """        Monotonic piecewise linear function f_i(x_map).                Need to find the optimal values for beta (breakpoints) and alpha            alpha = (a~, a_0, a_1, ... , a_sigma)            beta =  (    b_0, b_1, ... , b_sigma)        where sigma+1 is the number of breakpoints.                To solve, we will first fix beta and solve for alpha (A alpha = y);        since A^T A is symeetric, can solve for alpha via cholesky        Then we will search for the optimal beta.                                Args:            x_map:      Sorted mapped values (m_0, .., m_V), where V+1 is the                         number of mapped values that F_i processes            y:          Indices of x_map (0, ... , V)                Returns:            alpha            beta        """        # Get parameters        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']        V_plus_1 = len(x_map)        psi = parameters.OPTIONS['est_avg_num_keys_shard']                # Initialize beta        beta = np.empty(sigma_plus_1)        beta[:] = np.NaN                beta[0] = x_map[0]        for i in range(1, sigma_plus_1):            ind = int(np.floor(i * (V_plus_1 - 1) / psi))            beta[i] = x_map[ind]                    # Find optimal Alpha for given breakpoint Beta        A  = self._form_A_matrix(x_map, y, beta)        alpha = self._cholesky_factorization(A, y)                        # Train                num_iterations = 1        while True:                        # Perform cell search along direction s to find learning rate  lr^(k) >=0            lr_new, loss_new, alpha_new, beta_new = self._compute_learning_rate(alpha, beta, x_map, y)                          # if num_iterations % 50 == 0:            #     print ("Iteration: %d - Error: %.4f" %(num_iterations, loss))            print("Iteration %s, Loss is %s" %(num_iterations, loss_new))                                    # Stopping Conditions            if (num_iterations == parameters.OPTIONS['max_iterations']):                print("Max iteration of %s reach. Terminating" %num_iterations)                break                            # if np.abs(sum(beta-beta_new)) < parameters.OPTIONS['tolerance']:            #     print("Converged")            #     break                        if lr_new == 0:                print("Failed to find new learning rate. Terminating.")                break                    num_iterations += 1            alpha = alpha_new            beta = beta_new                # Return the optimal alpha and beta        return alpha, beta        def _compute_learning_rate(self, alpha, beta, x_map, y):        """        Find a new learning rate lr>=0 that minmizes loss function, while still         satisfying sum a_i's non-negativity.                Perform a grid search over all possible hyperparameters                Returns:            learning_rate:      Returns new learning rate to use for k-th iteration        """                # Define the grid search parameters        learn_rates = parameters.OPTIONS['learning_rates']                        # Initialization for search        loss_best = self._square_loss_function(alpha, beta, x_map, y)        lr_best = 0  # Placeholder in case all learning_rates fail        beta_best = beta        alpha_best = alpha                # Compute step direction        A  = self._form_A_matrix(x_map, y, beta)        s = self._compute_s(alpha, A, beta, x_map, y)                for lr in learn_rates:            # See if learning_rate valid            try:                # Update beta: B^(k+1) = B^(k) + lr^(k)*s^(k)                # Ignore first entry of s vector (beta and alpha dimension are different)                beta_new = beta + lr * s[1:]                  valid1 = self._check_sum_a_non_negative(beta_new, x_map, y)                valid2 = self._check_beta_increasing(beta_new)                if not (valid1 and valid2):                    continue            except:                # Go back to beginning of for loop                print("Invalid learning rate")                continue                        # Find optimal Alpha for this particular breakpoint Beta_new            A  = self._form_A_matrix(x_map, y, beta_new)            alpha_new = self._cholesky_factorization(A, y)                        loss = self._square_loss_function(alpha_new, beta_new, x_map, y)                        if (loss < loss_best):                loss_best = loss                lr_best = lr                beta_best = beta_new                alpha_best = alpha_new                 return lr_best, loss_best, alpha_best, beta_best                                  def _compute_s(self, alpha, A, beta, x_map, y):        # Get parameters        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']        V_plus_1 = len(x_map)                    # Initialization: compute input matrices based on alpha & beta         K = np.diag(alpha)                G = np.empty((sigma_plus_1+1, V_plus_1))        G[:] = np.NaN        G[0,:] = -1        for row in range(1, sigma_plus_1+1):            b = beta[row - 1]            for col in range(V_plus_1):                x_i = x_map[col]                if x_i >= b:                    val = -1                else:                    val = 0                G[row, col] = val        # Form Y matrix which is positive definite (symmetric, positive pivots)        Y = 2 * K.dot(G).dot(np.transpose(G)).dot(np.transpose(K))                # Form g: g = 2*KGr, where r = A alpha - y        r = A.dot(alpha) - y        g = 2 * K.dot(G).dot(r)                # Update in direction s= -Y^{-1} g        try:            s = -1 * (np.linalg.inv(Y)).dot(g)        except:            print(beta)  #TODO: fix why Y is sometimes singular????            print(-1 * np.linalg.inv(Y))        return s                       def _check_beta_increasing(self, beta):        """        Check if a given beta vector is valid:            b_0 < b_1 < ... < b__sigma        """        prev = beta[0]        for i in range(1, len(beta)):            if prev > beta[i]:                return False            prev = beta[i]                    return True                    def _check_sum_a_non_negative(self, beta, x_map, y):        """         Check if a given learning_rate lr produces a beta that gives a valid alpha.        Must satisfy            sum_{i=0)^n a_i >= 0,    for all 0 <= n <= sigma                Returns            True or False:     Whether the sum of a_i's is non-negative        """                # Compute alpha        A  = self._form_A_matrix(x_map, y, beta)                try:            alpha = self._cholesky_factorization(A, y)        except:            return False                        # Verify alpha valid        for i in range(len(alpha)):            sum_a = np.sum(alpha[:i])                        if sum_a < 0:                return False                return True                    def _square_loss_function(self, alpha, beta, x, y):        """        Compute square loss function        L(alpha, beta) = \sum_{i=0}^V (f(x_i) - y_i)^2                Input:            alpha:      Vector alpha = (a~, a_0, a_1, ... , a_sigma)            beta:       Vector beta =  (    b_0, b_1, ... , b_sigma)            x:          Vector of mapped x values            y:          Vector of indices of x                Returns:            loss:       Square loss function        """                        # Iterate over all V+1 points        loss = 0        for i in range(len(x)):            f_i = self.piecewise_f_function(alpha, beta, x[i])            l = (f_i - y[i])**2            loss += l                return loss                def piecewise_f_function(self, alpha, beta, x_i):        """         Compute piecewise function f(x_i)        Input:            alpha:      Vector alpha = (a~, a_0, a_1, ... , a_sigma)            beta:       Vector beta =  (    b_0, b_1, ... , b_sigma)            x_i:        i-th value of x        Returns:            f:          f(x_i)        """        f = 0        if (beta[0] <= x_i) and (x_i <= beta[1]):            f += alpha[0] + alpha[1]*(x_i - beta[0])        for i in range(1, len(beta)):            if x_i >= beta[i]:                f += alpha[i+1]*(x_i - beta[i])            else:                break                    return f            def _form_A_matrix(self, x_map, y, beta):        """Given a fixd beta value, solve for alpha                 Returns:            A:      Matrix for least squares equation A alpha = y        """        # Get parameters        sigma_plus_1 = parameters.OPTIONS['num_breakpoints']        V_plus_1 = len(x_map)                # Generate A: (V+1) x (sigma+2) for row x col        A = np.empty((V_plus_1, sigma_plus_1+1))        A[:] = np.NaN        A[:, 0] = 1                # Create Indicator function        indicator = np.empty((V_plus_1, sigma_plus_1))        indicator[:] = np.NaN        indicator[:, 0] = 1                # Iterate across rows then cols        for row in range(0, V_plus_1):            x_i = x_map[row]            for col in range(1, sigma_plus_1):                b = beta[col]                if x_i >= b:                    indicator[row, col] = 1                else:                    indicator[row, col] = 0                # Fill in A        for row in range(0, V_plus_1):            x_i = x_map[row]            for col in range(1, sigma_plus_1+1):                b = beta[col-1]                                val = (x_i - b) * indicator[row, col-1]                A[row, col] = val                return A    def _cholesky_factorization(self, A, y):        """"        For least squares equation A alpha = y, solve for alpha.        Since A^T A is symeetric, can solve for alpha via cholesky factorization                Cholesky Factorization Algorithm: http://www.math.iit.edu/~fass/477577_Chapter_5.pdf, page 50                Input:            A:      Input matrix             y:      Vector                Returns:            alpha:  Vector solution to least squares equation A alpha = y        """        # Compute A^T A matrix        A_T = np.transpose(A)        LHS = A_T.dot(A)                # # Compute A^T y vector        RHS = A_T.dot(y)                # Compute choleksy decomposition: A^T A = R^T R                R = scipy.linalg.cho_factor(LHS)                # Cholesky factorization        alpha = scipy.linalg.cho_solve(R, RHS)                return alpha        """  ================== Helper Functions =========== """    # TODO: implement after train shard pred     def assign_new_shard_id(self, x):        """        After the Shard Prediction function SP is trained, assign a shard id to a         new point x \in R^d.            shard_id = floor(SP(m)),   where m is the mapped value of x                Input:            x:            Datapoint x \in R^d                    Output:            shard_id:     Shard id assigned to point x        """                # Compute mapped value        x_map = self.mapping_function(x)                # Compute shard prediction function        # sp = self.shard_pred_func(x_map) #TODO: complete this once you finish the shard prediction function!!        sp = 0                shard_id = np.floor(sp)                return shard_id                    # TODO: implement after train shard pred     def shard_prediction_classify_new(self, x_index, x_map):        """        Shard Prediction Function SP : R --> [0, +inf).                                                      A shard is the preimage of an interval [a,b) \subseteq [0, +inf) under        the mapping function M, i.e.            S = M^{-1} ([a, b])                                                      Input:            x_map:        the mapped value of the key x            x_index:      the index of the key x        Returns:            shard_id:   Unique shard_id  of the shard that the key x should be stored in                                                    """                return 0            def partition_mapped_values(self, U):        """        Compute a list of numbers M_p = [m_1*, ... , m_U*] that evenly partitions        the mapped values into U similar sized partitions, i.e. the number of         mapped values in [m_{i-1}*, m_i*] is almost the same for all 0 <= i <= U                Input:            U:          Number of partitions for the mapped values.                Returns:            m_p:        Number list that partitions the mapped values        """                list_mapped_values = self.mapped_values['mappped_value'].values.tolist()        partitions = np.array_split(list_mapped_values, U)                # M_p: take first index of each partition, plus last index        m_p = []        for val in partitions:            m_p.append(val[0])                    m_p.append(partitions[-1][-1])  #TODO: need to make it a little bigger? otherwise not included in range [m_i*, m_{i+1}*)]                return m_p, partitions        def create_mapped_values(self, dataset):        """        Compute the mapped values for all points in a dataset and save to a        pandas dataframe self.mapped_values.                Input:            dataset:    Input dataset (pandas dataframe) to generate mapped values from        """                # Create empty dataframe of the mapped values        index_mv = np.arange(0, len(dataset))        self.mapped_values = pandas.DataFrame(np.nan, index=index_mv,                                              columns=['mappped_value'])                        # Compute + save mapped values        for index, row in dataset.iterrows():            map_value = self.mapping_function(row)            self.mapped_values['mappped_value'][index] = map_value                # Sort mapped values        self.mapped_values = self.mapped_values.sort_values(by=['mappped_value'])            def mapping_function(self, x):        """        Monotonic mapping function M(x) that takes an input data point x=(x_0, x_1)        and maps it to 0 \cup R^{+}, i.e. to [0, +inf)            M = i + (mu(H_i)) / (mu(C_i))                    where cell C_i is the cell that contains x s.t.                    C_i = [l_0, u_0] x [l_1, u_1]               cell H_i = [l_0, x_0] x [l_1, x_1], where l_0 and l_1 are the lower bounds of C_i              mu() is the Lebesgue measure function.        Input:            x:      Input data point in form pandas dataframe????                    Returns            :       Mapping M(x)        """        # TODO: check each x_i >=0        for x_i in x:            assert x_i >= 0                # Get cell C_i that contains x        cell_num = self.get_cell_num_contains_x(x)        cell = self.grid_cells[cell_num]                # Create cell H_i        H_cell = self.get_grid_cell(cell_num)   #Copy list problems, so for now just recompute        for j in range(len(H_cell)):            H_cell[j][1] = x[j]                    # Lebesgue Measure        mu_h = self.lebesgue_measure(H_cell)        mu_c = self.lebesgue_measure(cell)            # Mapping Function                   M = cell_num  + mu_h / mu_c                      return M           def lebesgue_measure(self, cell):        """Compute lebesgue measure for a cell. Since we only have one point, x,        in our subset, this is just the area/volume of the cell                Returns:            volume:      Lesbesgue measure of given cell        """                interval_lengths = []        for interval in cell:            length = interval[-1] - interval[0]            interval_lengths.append(length)                    volume = np.prod(interval_lengths)        return volume                    def get_cell_num_contains_x(self, x):        partition_indices = []                # Check along each axis in border_points        dim = len(parameters.OPTIONS['datasets']['labels'])        for i in range(dim):            col_name = parameters.OPTIONS['datasets']['labels'][i]            axis = self.border_points[col_name]            found = False                        for j in range(0, len(axis)-1):                if x[col_name] <= axis[j+1]:                    partition_indices.append(j)                    found = True                    break            if found == False:                raise ValueError("Input datapoint not within borderpoints along %s axis" %col_name)                       # Determine grid cell number based on partition indices        # t = T_1*j_0 + j_1 for cell C_t        cell_num = parameters.OPTIONS['num_partitions'][1] * partition_indices[0] + partition_indices[1]                return cell_num                    def generate_grid_cells(self):        """"        Generate all grid cells based on self.border_points and saves them in self.grid_cells.                self.grid_cells in format [C_0, C_1, C_2, ...]                          """        # Number of cells = T_1 * ... * T_{d-1}        num_cells = np.prod(parameters.OPTIONS['num_partitions'])                for i in range(num_cells):            grid_cell = self.get_grid_cell(i)            self.grid_cells.append(grid_cell)                 def get_grid_cell(self, cell_num):            """            A grid cell is define by its lower and upper bounds for each axis.             i.e. C = [l_0, u_0) x [l_1, u_1) x ... x [l_{d-1}, u_{d-1})                                                                  Cell numbering is C_0, C_1, ... C_{T_1 x T_2 ... T_{d-1}} which is            organized (for visualization, in 2D).                        For simplicity, we are only doing it in 2 dimensions, thus the formula is                  t = T_1*j_0 + j_1 for cell C_t            where t     : cell number                  T_i   : number of partitions in axis i                  j_i   : Upper bound/border for cell C_T along  axis i                        # Again, for simplicity, T_i = T_j                        ^ X_1            |            |            C_2     C_5            C_1     C_4            C_0     C_3 ______> X_0                        Returns:                grid_cell:      Grid cell in format [[l_0, u_0], [l_1, u_1]]                                    """            dim = len(parameters.OPTIONS['datasets']['labels'])            T = parameters.OPTIONS['num_partitions']            grid_cell = []            j=[]                        # j_0 = cell_num // T_1            j_0 = cell_num // T[1]            j.append(j_0)            # print("cell_num is: %s" %cell_num)            # print("j_0 is: %s" %j_0)                        # j_1 = t - (T_1 * j_0)            j_1 = cell_num - ( T[1] * j_0)            j.append(j_1)            # print("j_1 is: %s" %j_1)                        # Grab lower & upper bounds along each axis            for i in range(dim):                axis = parameters.OPTIONS['datasets']['labels'][i]                # lb                lb = self.border_points[axis][j[i]]                                                # ub                ub = self.border_points[axis][j[i]+1]                                # Add bounds to cell                grid_cell.append([lb, ub])                        return grid_cell                                        def generate_border_points(self, dataframe):        """        Partition the space into a series of grid cells based on the data         distribution along a sequence of axes and numbering the cells along these axes.                Stores the border points generated by the partition operation in        self.border_points.        Args:            dataframe:      Input dataset (pandas dataframe)        """                dim = len(parameters.OPTIONS['datasets']['labels'])                        # For each axis x_i        for  i  in range(dim):            # Partition into T_i parts; save only border_points            col_name = parameters.OPTIONS['datasets']['labels'][i]            self.partition_1D_dataframe(parameters.OPTIONS['num_partitions'][i],                                        col_name,                                        dataframe[col_name])                def partition_1D_dataframe(self, num_partitions, col_name, dataframe):        """        Given a 1D Pandas dataframe, partition into num_partitions of similar size.        Save the border points in self.border_points for col_name        Args:            num_partitions:         Number of partitions            col_name:               Name of column/axis to conduct partition on            dataframe:              Input 1D pandas dataframe        """        # Sort points        sort = dataframe.sort_values()                # Partition into T_i parts        chunk_size = math.ceil(len(sort) / num_partitions)  #TODO: fix if not perfect divide??                        # Lower Bound: origin        self.border_points[col_name].iloc[0] = 0                # Internal Points: take midpoint of extremes of adjacent paritions        for i in range(1, num_partitions):            lower = sort.iloc[i * chunk_size-1]                        # TODO: very hacky fix for not perfect divide (look up np.array_split)            if i == num_partitions:                upper = sort.iloc[-1]            else:                                upper = sort.iloc[i * chunk_size]                        border = (lower + upper) / 2            self.border_points[col_name].iloc[i] = border                # Upper Bound: ceiling of last point        ub = math.ceil(sort.iloc[-1])        self.border_points[col_name].iloc[-1] = ubif __name__ == '__main__':            """ 0)  Download Dataset """    initial_dataset, extra_dataset = utils.load_dataset()              ''' ###################### Construction of Index ###################### '''    index = LISA(initial_dataset, parameters.OPTIONS['page_size'])                # TODO: remove once done testing    x = initial_dataset.iloc[100]    print(x)    # test = index.mapped_values    # x_map = index.mapping_function(x)        m_p, partitions = index.partition_mapped_values(10)                         ''' ###################### Query Processing ###################### '''    # Range Queries                     # KNN Queries             ''' ###################### Create Visualizations for Performance ###################### '''