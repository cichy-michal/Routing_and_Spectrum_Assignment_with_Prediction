import os

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
        for iteration_file in os.listdir(self.folder_path):
            
            iteration_path = os.path.join(self.folder_path, iteration_file)
            iteration_requests = []
            
            with open(iteration_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    source = int(parts[0])
                    destination = int(parts[1])
                    bitrate = float(parts[2])
                    duration = int(parts[3])
                    iteration_requests.append((
                        source,
                        destination,
                        bitrate,
                        duration
                    ))
            
            self.iterations.append((
                iteration_file,
                iteration_requests
            ))
        
        print(f"Załadowano instancję {self.folder_name}: {len(self.iterations)} iteracji")

    def get_total_requests_count(self):
        total = 0
        for iteration in self.iterations:
            total += len(iteration[1])
        return total


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

if __name__ == "__main__":
    print("Wczytywanie danych sieci")
    
    cities = load_cities('lista miast.txt')
    topology, num_nodes, num_links = load_topology('ff.net')
    instances = load_instances('Problem routingu - dane')
    
    print_network_info(cities, topology)
    if instances:
        print_instances_info(instances)