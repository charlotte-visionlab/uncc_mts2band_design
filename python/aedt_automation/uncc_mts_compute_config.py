import socket

# TODO: Configure the scratch directory for each computer
# TODO: Configure the path to Ansys for each computer
class SolverConfig:
    compute_database = {}
    solver_configuration_default_pc = {
        "machine": "localhost",
        "num_cores": 16,
        "num_tasks": 1,
        "num_gpu": 1
    }
    solver_configuration_visionlab35_pc = {
        "machine": "visionlab35-pc",
        "num_cores": 16,  # (8 cores total, 64 GB RAM)
        "num_tasks": 1,
        "num_gpu": 1
    }
    solver_configuration_ece_emag1 = {
        "machine": "ece-emag1",
        "num_cores": 40,  # (20 cores total, 256 GB RAM)
        "num_tasks": 1,
        "num_gpu": 1
    }
    solver_configuration_ece_emag2 = {
        "machine": "ece-emag2",
        "num_cores": 48,  # (24 cores total, 512 GB RAM)
        "num_tasks": 1,
        "num_gpu": 1
    }
    solver_configuration_ece_emag3 = {
        "machine": "ece-emag3",
        "num_cores": 48,  # (24 cores total, 512 GB RAM)
        "num_tasks": 1,
        "num_gpu": 1
    }
    compute_database["default"] = solver_configuration_default_pc
    compute_database["visionlab35-pc"] = solver_configuration_visionlab35_pc
    compute_database["ece-emag1"] = solver_configuration_ece_emag1
    compute_database["ece-emag2"] = solver_configuration_ece_emag2
    compute_database["ece-emag3"] = solver_configuration_ece_emag3

    def __init__(self):
        self.hostname = socket.gethostname()
        if self.hostname in SolverConfig.compute_database.keys():
            self.solver_config = SolverConfig.compute_database[self.hostname]
        else:
            self.solver_config = SolverConfig.compute_database["default"]
        self.machine = self.solver_config["machine"]
        self.num_cores = self.solver_config["num_cores"]
        self.num_tasks = self.solver_config["num_tasks"]
        self.num_gpu = self.solver_config["num_gpu"]

    def __str__(self):
        return self.hostname
