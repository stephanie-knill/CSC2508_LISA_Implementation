"""Created on Tues Dec 29 2020@author: stephanieKnill"""#import osimport numpy as npimport mathimport pandas# Modulesimport parametersimport query_processingimport visualizationsimport utils# 3rd Part Modulesimport lebesgueclass LISA():    """    Defines a Learned Index Structure for Spatial Data (LISA)        Args:        keys_init:          Keys in the initial dataset        page_size:          Predefined page size            Attributes:        keys_init:          Keys in the initial dataset        page_size:          Predefined page size        map_func:           Partially monotonic mapping function M        shard_pred_func:    Shard prediction function         local_models:       Local models for all shards    """            def __init__(self, keys_init, page_size):        self.keys_init = keys_init        self.page_size = page_size                # Create empty dataframe of border_points        index = np.arange(0, parameters.OPTIONS['num_partitions'][0] + 1)           #TODO: for now simplify 10 dataset 2 dimensions, 2) num partitions T_i = T_j        self.border_points = pandas.DataFrame(np.nan, index=index,                                               columns=parameters.OPTIONS['datasets']['labels'])        # Create empty list of grid cells        self.grid_cells = []                # Gogogogo!        self.build_LISA()                # TODO: remove after testing        self.initial_dataset                    def build_LISA(self):        """ Build LISA  """                # 0)  Download Dataset        initial_dataset, extra_dataset = utils.load_dataset()                  # TODO: remove after testing        self.initial_dataset = initial_dataset        # 1) Generation of Grid Cells        # Generate border points        self.generate_border_points(initial_dataset)                print(self.border_points)                # Generate all grid cells        self.generate_grid_cells()        # self.get_grid_cell(0)                                # 2) Partially Monotonic Mapping Function M                                # 3) Monotonic Shard Prediction Function SP                        # 4) Construction of Local Models L        return 0            def mapping_function(self, x):        """        Monotonic mapping function M(x) that takes an input data point x=(x_0, x_1).            M = i + (mu(H_i)) / (mu(C_i))                    where cell C_i is the cell that contains x s.t.                    C_i = [l_0, u_0] x [l_1, u_1]               cell H_i = [l_0, x_0] x [l_1, x_1], where l_0 and l_1 are the lower bounds of C_i              mu() is the Lebesgue measure function.        Input:            x:      Input data point in form pandas dataframe????                    Returns            :       Mapping M(x)        """        # TODO: check each x_i >=0        for x_i in x:            assert x_i >= 0                # Get cell C_i that contains x        cell_num = self.get_cell_num_contains_x(x)        cell = self.get_grid_cell(cell_num)                # Create cell H_i        #TODO!!!!                        # lebesgue.lebesgue_function( n, x, nfun, xfun ):        return 0                            def get_cell_num_contains_x(self, x):        partition_indices = []                # Check along each axis in border_points        dim = len(parameters.OPTIONS['datasets']['labels'])        for i in range(dim):            col_name = parameters.OPTIONS['datasets']['labels'][i]            axis = self.border_points[col_name]            found = False                        for j in range(0, len(axis)-1):                if x[col_name] <= axis[j+1]:                    partition_indices.append(j)                    found = True                    break            if found == False:                raise Error("Input datapoint not within borderpoints along %s axis" %col_name)                       # Determine grid cell number based on partition indices        # t = T_1*j_0 + j_1 for cell C_t        cell_num = parameters.OPTIONS['num_partitions'][1] * partition_indices[0] + partition_indices[1]                return cell_num                            def generate_grid_cells(self):        """"        Generate all grid cells based on self.border_points and saves them in self.grid_cells.                self.grid_cells in format [C_0, C_1, C_2, ...]                          """        # Number of cells = T_1 * ... * T_{d-1}        num_cells = np.prod(parameters.OPTIONS['num_partitions'])                for i in range(num_cells):            grid_cell = self.get_grid_cell(i)            self.grid_cells.append(grid_cell)                        def get_grid_cell(self, cell_num):            """            A grid cell is define by its lower and upper bounds for each axis.             i.e. C = [l_0, u_0) x [l_1, u_1) x ... x [l_{d-1}, u_{d-1})                                                                  Cell numbering is C_0, C_1, ... C_{T_1 x T_2 ... T_{d-1}} which is            organized (for visualization, in 2D).                        For simplicity, we are only doing it in 2 dimensions, thus the formula is                  t = T_1*j_0 + j_1 for cell C_t            where t     : cell number                  T_i   : number of partitions in axis i                  j_i   : Upper bound/border for cell C_T along  axis i                        # Again, for simplicity, T_i = T_j                        ^ X_1            |            |            C_2     C_5            C_1     C_4            C_0     C_3 ______> X_0                        Returns:                grid_cell:      Grid cell in format [[l_0, u_0], [l_1, u_1]]                                    """            dim = len(parameters.OPTIONS['datasets']['labels'])            T = parameters.OPTIONS['num_partitions']            grid_cell = []            j=[]                        # j_0 = cell_num // T_1            j_0 = cell_num // T[1]            j.append(j_0)            # print("cell_num is: %s" %cell_num)            # print("j_0 is: %s" %j_0)                        # j_1 = t - (T_1 * j_0)            j_1 = cell_num - ( T[1] * j_0)            j.append(j_1)            # print("j_1 is: %s" %j_1)                        # Grab lower & upper bounds along each axis            for i in range(dim):                axis = parameters.OPTIONS['datasets']['labels'][i]                # lb                lb = self.border_points[axis][j[i]]                                                # ub                ub = self.border_points[axis][j[i]+1]                                # Add bounds to cell                grid_cell.append([lb, ub])                        return grid_cell                                        def generate_border_points(self, dataframe):        """        Partition the space into a series of grid cells based on the data         distribution along a sequence of axes and numbering the cells along these axes.                Stores the border points generated by the partition operation in        self.border_points.        Args:            dataframe:      Input dataset (pandas dataframe)        """                dim = len(parameters.OPTIONS['datasets']['labels'])                        # For each axis x_i        for  i  in range(dim):            # Partition into T_i parts; save only border_points            col_name = parameters.OPTIONS['datasets']['labels'][i]            self.partition_1D_dataframe(parameters.OPTIONS['num_partitions'][i],                                        col_name,                                        dataframe[col_name])                def partition_1D_dataframe(self, num_partitions, col_name, dataframe):        """        Given a 1D Pandas dataframe, partition into num_partitions of similar size.        Save the border points in self.border_points for col_name        Args:            num_partitions:         Number of partitions            col_name:               Name of column/axis to conduct partition on            dataframe:              Input 1D pandas dataframe        """        # Sort points        sort = dataframe.sort_values()                # Partition into T_i parts        chunk_size = math.ceil(len(sort) / num_partitions)  #TODO: fix if not perfect divide??                        # Lower Bound: origin        self.border_points[col_name].iloc[0] = 0                # Internal Points: take midpoint of extremes of adjacent paritions        for i in range(1, num_partitions):            lower = sort.iloc[i * chunk_size-1]                        # TODO: very hacky fix for not perfect divide            if i == num_partitions:                upper = sort.iloc[-1]            else:                                upper = sort.iloc[i * chunk_size]                        border = (lower + upper) / 2            self.border_points[col_name].iloc[i] = border                # Upper Bound: ceiling of last point        ub = math.ceil(sort.iloc[-1])        self.border_points[col_name].iloc[-1] = ubif __name__ == '__main__':            ''' ###################### Construction of Index ###################### '''    index = LISA(parameters.OPTIONS['keys_init'], parameters.OPTIONS['page_size'])            x = index.initial_dataset.iloc[100]    print(x)    # index.mapping_function(x)                         ''' ###################### Query Processing ###################### '''    # Range Queries                     # KNN Queries             ''' ###################### Create Visualizations for Performance ###################### '''