import os
import heapq
import math
import csv
from sklearn.tree import DecisionTreeRegressor

class Instance:
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.folder_name = os.path.basename(folder_path)
        self.traffic_value = None
        self.case_id = None
        self.iterations = []
        self.parse_folder_name()
        self.load_requests()
    
    def parse_folder_name(self):
        traffic_value_str, case_id_str = self.folder_name.split('_')
        self.traffic_value = int(traffic_value_str)
        self.case_id = int(case_id_str)
        
    def load_requests(self):
        iteration_files = sorted(
            os.listdir(self.folder_path),
            key=lambda name: int(os.path.splitext(name)[0])
        )
        self.iterations = []
        for iteration_file in iteration_files:
            iteration_path = os.path.join(self.folder_path, iteration_file)
            iteration_requests = []
            with open(iteration_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    source = int(parts[0])
                    destination = int(parts[1])
                    bitrate = float(parts[2])
                    duration = int(parts[3])
                    iteration_requests.append(
                        (source, destination, bitrate, duration)
                    )
            self.iterations.append(
                (iteration_file, iteration_requests)
            )
        print(f"Załadowano instancję {self.folder_name}: {len(self.iterations)} iteracji")

    def get_total_requests_count(self):
        total = 0
        for _, iteration_requests in self.iterations:
            total += len(iteration_requests)
        return total

class ElasticOpticalNetwork:
    def __init__(self, cities, topology, num_nodes, num_links):
        self.cities = cities
        self.topology = [row[:] for row in topology]
        self.base_topology = [row[:] for row in self.topology]
        self.num_nodes = num_nodes
        self.num_links = num_links
        self.num_fs = 320
        self.city_indices = {}
        for index, city in enumerate(cities):
            name = city[0]
            self.city_indices[name.lower()] = index
        self.spectrum_state = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.topology[i][j] > 0:
                    self.spectrum_state[(i, j)] = [0] * self.num_fs
        self.active_connections = {}
        self.connection_counter = 1
        self.modulation_formats = {
        1: {'name': 'BPSK', 'reach': 6300, 'level': 50},
        2: {'name': 'QPSK', 'reach': 3500, 'level': 100},
        3: {'name': '16-QAM', 'reach': 1200, 'level': 150},
        4: {'name': '32-QAM', 'reach': 600, 'level': 200}
        }
        self.predictor = TrafficPredictor(num_links)
        self.current_iteration = 0
        self.link_map = {}
        self.link_used = {}
        for link in self.spectrum_state:
            self.link_used[link] = 0
        self.total_used = 0
        self.link_map_reverse = {}
        idx = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.topology[i][j] > 0:
                    self.link_map[(i, j)] = idx
                    self.link_map_reverse[idx] = (i, j)
                    idx += 1
    
    def dijkstra(self, source, target):
        dist = [float('inf')] * self.num_nodes
        prev = [-1] * self.num_nodes
        dist[source] = 0
        priority_queue = [(0, source)]
        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            if current_node == target:
                break
            if current_dist > dist[current_node]:
                continue
            for node in range(self.num_nodes):
                if self.topology[current_node][node] > 0:
                    alt = dist[current_node] + self.topology[current_node][node]
                    if alt < dist[node]:
                        dist[node] = alt
                        prev[node] = current_node
                        heapq.heappush(priority_queue, (alt, node))
        path = []
        current_node = target
        while current_node != -1:
            path.append(current_node)
            current_node = prev[current_node]
        return path[::-1] if path and path[-1] == source else []
    
    def k_shortest_paths(self, source, target, k=3):
        shortest_path = self.dijkstra(source, target)
        if not shortest_path:
            return []
        paths = [shortest_path]
        candidates = []
        for ki in range(1, k):
            for i in range(len(paths[ki-1]) - 1):
                spur_node = paths[ki-1][i]
                root_path = paths[ki-1][:i+1]
                temp_topology = [row[:] for row in self.topology]
                for path in paths:
                    if len(path) > i + 1 and path[:i+1] == root_path:
                        u = path[i]
                        v = path[i+1]
                        temp_topology[u][v] = 0
                spur_path = self.dijkstra_with_topology(temp_topology, spur_node, target)
                if spur_path:
                    total_path = root_path + spur_path[1:]
                    if total_path not in paths and total_path not in [c[1] for c in candidates]:
                        path_length = self.calculate_path_length(total_path)
                        heapq.heappush(candidates, (path_length, total_path))
            if not candidates:
                break
            _, next_path = heapq.heappop(candidates)
            paths.append(next_path)
        return paths[:k]
    
    def dijkstra_with_topology(self, topology, source, target):
        num_nodes = len(topology)
        dist = [float('inf')] * num_nodes
        prev = [-1] * num_nodes
        dist[source] = 0
        priority_queue = [(0, source)]
        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            if current_node == target:
                break
            if current_dist > dist[current_node]:
                continue
            for node in range(num_nodes):
                if topology[current_node][node] > 0:
                    alt = dist[current_node] + topology[current_node][node]
                    if alt < dist[node]:
                        dist[node] = alt
                        prev[node] = current_node
                        heapq.heappush(priority_queue, (alt, node))
        path = []
        current_node = target
        while current_node != -1:
            path.append(current_node)
            current_node = prev[current_node]
        return path[::-1] if path and path[-1] == source else []
    
    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.topology[path[i]][path[i+1]]
        return length
    
    def select_modulation_format(self, path_length):
        for level in sorted(self.modulation_formats.keys(), reverse=True):
            if path_length <= self.modulation_formats[level]['reach']:
                return level
        return 1
    
    def calculate_required_fs(self, bitrate, modulation_format):
        capacity_per_3_slots = self.modulation_formats[modulation_format]['level']
        num_slot_groups = math.ceil(bitrate / capacity_per_3_slots)
        N_s = num_slot_groups * 3
        return N_s
    
    def check_spectrum_availability(self, path, start_fs, num_fs):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = (u, v)
            if start_fs + num_fs > self.num_fs:
                return False
            for slot in range(start_fs, start_fs + num_fs):
                if self.spectrum_state[link][slot] != 0:
                    return False
        return True
    
    def allocate_spectrum(self, path, start_fs, num_fs, connection_id):
        end_fs = start_fs + num_fs
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = (u, v)
            slots = self.spectrum_state[link]
            for slot in range(start_fs, end_fs):
                slots[slot] = connection_id
            self.link_used[link] += num_fs
            self.total_used += num_fs
    
    def release_spectrum(self, connection_id):
        for link in self.spectrum_state:
            for slot in range(self.num_fs):
                if self.spectrum_state[link][slot] == connection_id:
                    self.spectrum_state[link][slot] = 0
    
    def release_connection(self, connection_id):
        conn = self.active_connections[connection_id]
        path = conn['path']
        start_fs = conn['start_fs']
        num_fs = conn['num_fs']
        end_fs = start_fs + num_fs
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = (u, v)
            slots = self.spectrum_state[link]
            for slot in range(start_fs, end_fs):
                slots[slot] = 0
            self.link_used[link] -= num_fs
            self.total_used -= num_fs
    
    def find_available_blocks(self, path, required_fs):
        available_blocks = []
        common_availability = [1] * self.num_fs
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = (u, v)
            for slot in range(self.num_fs):
                if self.spectrum_state[link][slot] != 0:
                    common_availability[slot] = 0
        current_block_start = -1
        for slot in range(self.num_fs):
            if common_availability[slot] == 1:
                if current_block_start == -1:
                    current_block_start = slot
            else:
                if current_block_start != -1:
                    block_size = slot - current_block_start
                    if block_size >= required_fs:
                        available_blocks.append((current_block_start, slot - 1))
                    current_block_start = -1
        if current_block_start != -1:
            block_size = self.num_fs - current_block_start
            if block_size >= required_fs:
                available_blocks.append((current_block_start, self.num_fs - 1))
        return available_blocks
    
    def calculate_quality_score(self, required_fs, block_size):
        return 1 - (required_fs + 1) / block_size
    
    def phase1_strict_exact_fit(self, path, required_fs):
        for start_fs in range(self.num_fs - required_fs + 1):
            if not self.check_spectrum_availability(path, start_fs, required_fs):
                continue
            left_occupied = True
            if start_fs != 0:
                left_idx = start_fs - 1
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    if self.spectrum_state[(u, v)][left_idx] == 0:
                        left_occupied = False
                        break
            right_occupied = True
            right_idx = start_fs + required_fs
            if right_idx != self.num_fs:
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    if self.spectrum_state[(u, v)][right_idx] == 0:
                        right_occupied = False
                        break
            if left_occupied and right_occupied:
                return start_fs
        return -1

    def pfa_sm_rsa_algorithm(self, source, destination, bitrate, use_prediction):
        old_costs = {}
        if use_prediction and self.predictor.is_trained:
            predicted_loads = self.predictor.predict(self.predictor.link_loads_history)
            if predicted_loads is not None:
                current_loads = self.get_link_loads()
                for idx, pred_load in enumerate(predicted_loads):
                    if idx not in self.link_map_reverse:
                        continue
                    u, v = self.link_map_reverse[idx]
                    now_load = current_loads[idx]
                    score = 0.5 * pred_load + 0.5 * now_load
                    factor = 1.0
                    if score > 0.95:
                        factor = 3.0
                    elif score > 0.90:
                        factor = 2.0
                    elif score > 0.85:
                        factor = 1.5
                    if factor != 1.0:
                        if (u, v) not in old_costs:
                            old_costs[(u, v)] = self.topology[u][v]
                        base_cost = self.base_topology[u][v]
                        self.topology[u][v] = int(base_cost * factor)
        k_paths = self.k_shortest_paths(source, destination, k=3)
        for (u, v), old_cost in old_costs.items():
            self.topology[u][v] = old_cost
        if not k_paths:
            return False, None, None, None
        for path in k_paths:
            path_length = self.calculate_path_length(path)
            modulation_level = self.select_modulation_format(path_length)
            required_fs = self.calculate_required_fs(bitrate, modulation_level)
            start_fs = self.phase1_strict_exact_fit(path, required_fs)
            if start_fs != -1:
                return True, path, start_fs, required_fs
            available_blocks = self.find_available_blocks(path, required_fs)
            if not available_blocks:
                continue
            ideal_blocks = []
            non_zero_blocks = []
            zero_blocks = []
            for block_start, block_end in available_blocks:
                block_size = block_end - block_start + 1
                quality_score = self.calculate_quality_score(required_fs, block_size)
                if abs(quality_score - (-1 / block_size)) < 1e-6:
                    ideal_blocks.append((block_start, block_end, quality_score))
                elif quality_score > 0:
                    non_zero_blocks.append((block_start, block_end, quality_score))
                elif abs(quality_score) < 1e-6:
                    zero_blocks.append((block_start, block_end, quality_score))
            if ideal_blocks:
                block_start, block_end, _ = ideal_blocks[0]
                return True, path, block_start, required_fs
            if non_zero_blocks:
                non_zero_blocks.sort(key=lambda x: x[2])
                block_start, block_end, _ = non_zero_blocks[0]
                return True, path, block_start, required_fs
            if zero_blocks:
                block_start, block_end, _ = zero_blocks[0]
                return True, path, block_start, required_fs
        return False, None, None, None

    
    def ff_rsa_algorithm(self, source, destination, bitrate):
        path = self.dijkstra(source, destination)
        if not path:
            return False, None, None, None
        path_length = 0.0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            path_length += self.topology[u][v]
        modulation_format = self.select_modulation_format(path_length)
        required_fs = self.calculate_required_fs(bitrate, modulation_format)
        for start_fs in range(0, self.num_fs - required_fs + 1):
            if self.check_spectrum_availability(path, start_fs, required_fs):
                return True, path, start_fs, required_fs
        return False, None, None, None

    def process_request_pfa(self, source, destination, bitrate, duration):
        success, path, start_fs, num_fs = self.pfa_sm_rsa_algorithm(source, destination, bitrate, use_prediction=False)
        if success:
            connection_id = self.connection_counter
            self.connection_counter += 1
            self.allocate_spectrum(path, start_fs, num_fs, connection_id)
            self.active_connections[connection_id] = {
                'path': path, 
                'start_fs': start_fs, 
                'num_fs': num_fs,
                'remaining_time': duration, 
                'bitrate': bitrate
            }
            return True, connection_id
        return False, None
    
    def process_request_pred(self, source, destination, bitrate, duration):
        success, path, start_fs, num_fs = self.pfa_sm_rsa_algorithm(source, destination, bitrate,use_prediction=True)
        if success:
            connection_id = self.connection_counter
            self.connection_counter += 1
            self.allocate_spectrum(path, start_fs, num_fs, connection_id)
            self.active_connections[connection_id] = {
                'path': path,
                'start_fs': start_fs,
                'num_fs': num_fs,
                'remaining_time': duration,
                'bitrate': bitrate
            }
            return True, connection_id
        return False, None
    
    def process_request_ff(self, source, destination, bitrate, duration):
        success, path, start_fs, num_fs = self.ff_rsa_algorithm(source, destination, bitrate)
        if success:
            connection_id = self.connection_counter
            self.connection_counter += 1
            self.allocate_spectrum(path, start_fs, num_fs, connection_id)
            self.active_connections[connection_id] = {
                'path': path,
                'start_fs': start_fs,
                'num_fs': num_fs,
                'remaining_time': duration,
                'bitrate': bitrate
            }
            return True, connection_id
        return False, None
    
    def update_network_state(self):
        completed_connections = []
        for conn_id, conn_data in self.active_connections.items():
            conn_data['remaining_time'] -= 1
            if conn_data['remaining_time'] <= 0:
                completed_connections.append(conn_id)
        for conn_id in completed_connections:
            self.release_connection(conn_id)
            del self.active_connections[conn_id]
    
    def calculate_spectrum_utilization(self):
        total_slots = len(self.spectrum_state) * self.num_fs
        if total_slots == 0:
            return 0.0
        return self.total_used / total_slots
    
    def get_link_loads(self):
        loads = [0.0] * self.num_links
        for link, idx in self.link_map.items():
            loads[idx] = self.link_used[link] / self.num_fs
        return loads
    
    def train_predictor(self):
        self.predictor.train()

class TrafficPredictor:
    def __init__(self, num_links):
        self.model = DecisionTreeRegressor()
        self.history_X = []
        self.history_y = []
        self.is_trained = False
        self.num_links = num_links
        self.window_size = 4
        self.link_loads_history = [] 

    def update_history(self, current_link_loads):
        self.link_loads_history.append(current_link_loads)
        if len(self.link_loads_history) >= self.window_size + 1:
            features = []
            past = self.link_loads_history[-(self.window_size + 1):-1]
            for vec in past:
                features.extend(vec)
            target = current_link_loads
            self.history_X.append(features)
            self.history_y.append(target)
    def train(self):
        if len(self.history_X) > 10:
            self.model.fit(self.history_X, self.history_y)
            self.is_trained = True

    def predict(self, recent_history):
        if (not self.is_trained) or (len(recent_history) < self.window_size):
            return None
        features = []
        start = len(recent_history) - self.window_size
        for i in range(start, start + self.window_size):
            features.extend(recent_history[i])
        prediction = self.model.predict([features])
        return prediction[0]

def load_cities(cities_file):
    cities = []
    with open(cities_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                name = parts[1]
                country = parts[2]
                cities.append((name, country))
    print(f"Załadowano {len(cities)} miast")
    return cities

def load_topology(topology_file):
    with open(topology_file, 'r') as f:
        lines = f.readlines()
    num_nodes = int(lines[0].strip())
    num_links = int(lines[1].strip())
    print(f"Liczba węzłów: {num_nodes}")
    print(f"Liczba łączy: {num_links}")
    topology = []
    for i in range(num_nodes):
        topology.append([0] * num_nodes)
    for i in range(num_nodes):
        distances = list(map(int, lines[i + 2].strip().split()))
        for j in range(num_nodes):
            topology[i][j] = distances[j]
    return topology, num_nodes, num_links

def load_instances(instances_directory):
    instances = []
    for folder_name in os.listdir(instances_directory):
        folder_path = os.path.join(instances_directory, folder_name)
        if os.path.isdir(folder_path):
            instance = Instance(folder_path)
            instances.append(instance)
    print(f"Załadowano {len(instances)} instancji")
    return instances

def print_network_info(cities, topology):
    print("\nLista miast:")
    for i, city in enumerate(cities):
        print(f"  {i}: {city[0]} ({city[1]})")
    print("\nPołączenia w sieci (odległości != 0):")
    connections = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            if topology[i][j] > 0:
                connections.append((cities[i][0], cities[j][0], topology[i][j]))
    for conn in connections:
        print(f"  {conn[0]} <-> {conn[1]}: {conn[2]} km")
    print(f"\nŁączna liczba połączeń dwukierunkowych: {len(connections)}")

def print_instances_info(instances):
    print("\nInformacje o instancjach")
    for instance in instances:
        total_requests = instance.get_total_requests_count()
        print(f"  {instance.folder_name}: {len(instance.iterations)} iteracji, {total_requests} żądań")

def save_bp_csv(filename, rows, append=False):
    mode = "a" if append else "w"
    with open(filename, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)

def simulate_instance(network, instance):
    statistics = {
        'total_requests': 0,
        'blocked_requests': 0,
        'spectrum_utilization': []
    }
    bp_rows = []
    bp_csv = f"bp_Pred-PFA-SM-RSA_{instance.folder_name}.csv"
    print(f"Rozpoczynanie symulacji ((PFA-SM-RSA_PRED)) dla instancji {instance.folder_name}")
    for i, iteration in enumerate(instance.iterations):
        iteration_file, requests = iteration
        network.current_iteration = i
        current_loads = network.get_link_loads()
        network.predictor.update_history(current_loads)
        if i > 100 and i % 20 == 0:
            network.train_predictor()
        for request in requests:
            source, destination, bitrate, duration = request
            statistics['total_requests'] += 1
            success, _ = network.process_request_pred(source, destination, bitrate, duration)
            if not success:
                statistics['blocked_requests'] += 1
        network.update_network_state()
        utilization = network.calculate_spectrum_utilization()
        statistics['spectrum_utilization'].append(utilization)
        if statistics['total_requests'] > 0:
            current_bp = statistics['blocked_requests'] / statistics['total_requests']
        else:
            current_bp = 0
        bp_rows.append([i + 1, current_bp])
    if statistics['total_requests'] > 0:
        statistics['blocking_probability'] = statistics['blocked_requests'] / statistics['total_requests']
    else:
        statistics['blocking_probability'] = 0
    if statistics['spectrum_utilization']:
        statistics['avg_spectrum_utilization'] = sum(statistics['spectrum_utilization']) / len(statistics['spectrum_utilization'])
    else:
        statistics['avg_spectrum_utilization'] = 0
    save_bp_csv(bp_csv, bp_rows, append=False)
    return statistics

def simulate_instance_ff(network, instance):
    statistics = {
        'total_requests': 0,
        'blocked_requests': 0,
        'spectrum_utilization': []
    }
    bp_rows = []
    bp_csv = f"bp_FF-RSA_{instance.folder_name}.csv"
    print(f"Rozpoczynanie symulacji (FF-RSA) dla instancji {instance.folder_name}")
    for i, iteration in enumerate(instance.iterations):
        iteration_file, requests = iteration
        network.current_iteration = i
        for request in requests:
            source, destination, bitrate, duration = request
            statistics['total_requests'] += 1
            success, _ = network.process_request_ff(source, destination, bitrate, duration)
            if not success:
                statistics['blocked_requests'] += 1
        network.update_network_state()
        utilization = network.calculate_spectrum_utilization()
        statistics['spectrum_utilization'].append(utilization)
        if statistics['total_requests'] > 0:
            current_bp = statistics['blocked_requests'] / statistics['total_requests']
        else:
            current_bp = 0
        bp_rows.append([i + 1, current_bp])
    if statistics['total_requests'] > 0:
        statistics['blocking_probability'] = (
            statistics['blocked_requests'] / statistics['total_requests']
        )
    else:
        statistics['blocking_probability'] = 0
    if statistics['spectrum_utilization']:
        statistics['avg_spectrum_utilization'] = (
            sum(statistics['spectrum_utilization']) /
            len(statistics['spectrum_utilization'])
        )
    else:
        statistics['avg_spectrum_utilization'] = 0
    save_bp_csv(bp_csv, bp_rows, append=False)
    return statistics

def simulate_instance_pfa(network, instance):
    statistics = {
        'total_requests': 0,
        'blocked_requests': 0,
        'spectrum_utilization': []
    }
    bp_rows = []
    bp_csv = f"bp_PFA-SM-RSA_{instance.folder_name}.csv"
    print(f"Rozpoczynanie symulacji (PFA-SM-RSA) dla instancji {instance.folder_name}")
    for i, iteration in enumerate(instance.iterations):
        iteration_file, requests = iteration
        network.current_iteration = i
        for request in requests:
            source, destination, bitrate, duration = request
            statistics['total_requests'] += 1
            success, _ = network.process_request_pfa(source, destination, bitrate, duration)
            if not success:
                statistics['blocked_requests'] += 1
        network.update_network_state()
        utilization = network.calculate_spectrum_utilization()
        statistics['spectrum_utilization'].append(utilization)
        if statistics['total_requests'] > 0:
            current_bp = statistics['blocked_requests'] / statistics['total_requests']
        else:
            current_bp = 0
        bp_rows.append([i + 1, current_bp])
    if statistics['total_requests'] > 0:
        statistics['blocking_probability'] = statistics['blocked_requests'] / statistics['total_requests']
    else:
        statistics['blocking_probability'] = 0
    if statistics['spectrum_utilization']:
        statistics['avg_spectrum_utilization'] = sum(statistics['spectrum_utilization']) / len(statistics['spectrum_utilization'])
    else:
        statistics['avg_spectrum_utilization'] = 0
    save_bp_csv(bp_csv, bp_rows, append=False)
    return statistics

def format_pl(x):
    return f"{x:.4f}".replace(".", ",")

def save_e2_summary_csv(filename, instances, results_ff, results_pfa, results_pred):
    instances_sorted = sorted(instances, key=lambda inst: (inst.traffic_value, inst.case_id))
    rows = []
    rows.append(["Instancja", "FF-RSA", "PFA-SM-RSA", "Pred-PFA-SM-RSA"])
    for inst in instances_sorted:
        name = inst.folder_name
        bp_ff = results_ff[name]["blocking_probability"]
        bp_pfa = results_pfa[name]["blocking_probability"]
        bp_pred = results_pred[name]["blocking_probability"]
        rows.append([name, format_pl(bp_ff), format_pl(bp_pfa), format_pl(bp_pred)])
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)
    print(f"\nZapisano podsumowanie E2 do pliku: {filename}")

if __name__ == "__main__":
    import time

    print("Wczytywanie danych sieci")
    cities = load_cities('lista miast.txt')
    topology, num_nodes, num_links = load_topology('ff.net')
    instances = load_instances('Problem routingu - dane')

    print_network_info(cities, topology)
    if instances:
        print_instances_info(instances)

    results_ff = {}
    results_pfa = {}
    results_pred = {}
    
    for instance in instances:
        print("\n==========================================")
        print(f"Instancja: {instance.folder_name}")
        print("==========================================")

        # ------------ BASELINE: FF–RSA ------------
        network_ff = ElasticOpticalNetwork(cities, topology, num_nodes, num_links)
        start_time_ff = time.time()
        res_ff = simulate_instance_ff(network_ff, instance)
        end_time_ff = time.time()
        res_ff['execution_time'] = end_time_ff - start_time_ff
        results_ff[instance.folder_name] = res_ff

        print("  [FF-RSA]")
        print(f"    Czas wykonania: {res_ff['execution_time']:.2f} s")
        print(f"    Łączna liczba żądań: {res_ff['total_requests']}")
        print(f"    Zablokowane żądania: {res_ff['blocked_requests']}")
        print(f"    Prawdopodobieństwo blokady (BP): {res_ff['blocking_probability']:.4f}")
        print(f"    Średnie wykorzystanie widma: {res_ff['avg_spectrum_utilization']:.4f}")

        # ------ PFA–SM–RSA (bez predykcji) ------
        network_pfa = ElasticOpticalNetwork(cities, topology, num_nodes, num_links)
        start_time_pfa = time.time()
        res_pfa = simulate_instance_pfa(network_pfa, instance)
        end_time_pfa = time.time()

        res_pfa['execution_time'] = end_time_pfa - start_time_pfa
        results_pfa[instance.folder_name] = res_pfa

        print("  [PFA-SM-RSA]")
        print(f"    Czas wykonania: {res_pfa['execution_time']:.2f} s")
        print(f"    Łączna liczba żądań: {res_pfa['total_requests']}")
        print(f"    Zablokowane żądania: {res_pfa['blocked_requests']}")
        print(f"    Prawdopodobieństwo blokady (BP): {res_pfa['blocking_probability']:.4f}")
        print(f"    Średnie wykorzystanie widma: {res_pfa['avg_spectrum_utilization']:.4f}")

        # ------ PFA–SM–RSA + predykcja ------
        network_pred = ElasticOpticalNetwork(cities, topology, num_nodes, num_links)
        start_time_pred = time.time()
        res_pred = simulate_instance(network_pred, instance)
        end_time_pred = time.time()
        res_pred['execution_time'] = end_time_pred - start_time_pred
        results_pred[instance.folder_name] = res_pred

        print("  [Pred-PFA-SM-RSA]")
        print(f"    Czas wykonania: {res_pred['execution_time']:.2f} s")
        print(f"    Łączna liczba żądań: {res_pred['total_requests']}")
        print(f"    Zablokowane żądania: {res_pred['blocked_requests']}")
        print(f"    Prawdopodobieństwo blokady (BP): {res_pred['blocking_probability']:.4f}")
        print(f"    Średnie wykorzystanie widma: {res_pred['avg_spectrum_utilization']:.4f}")

    save_e2_summary_csv("E2_bp_summary.csv", instances, results_ff, results_pfa, results_pred)
