receptacle_categories = {
    "Chair": "Chr",
    "Sofa": "Sofa",
    "Table": "Tbl",
    "Tv Stand": "TvStnd",
    "Counter": "counter",
    "Drawer": "drawer",
    "Sink": "sink",
    "Fridge": "middle",
}


class Object:
    def __init__(self, name, receptacle, pos):
        self.name = name
        self.receptacle = receptacle
        self.pos = pos

        self.get_receptacle_category()
        self.get_short_name()
        self.inside_receptacle = self.get_accesibility()

    def get_short_name(self):
        self.short_name = " ".join(self.name.split("_")[1:-1])

    def get_receptacle_category(self):
        for k, v in receptacle_categories.items():
            if v in self.receptacle.split("_")[2]:
                self.receptacle_category = k
                return

    def set_position(self, rom, env):
        obj_id = rom.get_object_by_handle(self.name).object_id
        index = env.sim.scene_obj_ids.index(obj_id)
        scene_pos = env.sim.get_scene_pos()
        self.pos = scene_pos[index]

    def get_accesibility(self):
        if self.receptacle_category in ["Fridge", "Drawer"]:
            return False
        else:
            return True

    def __repr__(self) -> str:
        return f"{self.short_name} [{self.name}]"


class SceneParser:
    def __init__(self, sim):
        self.sim = sim
        self.rom = self.sim.get_rigid_object_manager()
        self.name_to_receptacle = self.sim.ep_info["name_to_receptacle"]
        self.scene_pos = self.sim.get_scene_pos()
        self.get_objects()
        self.group_objects_by_receptacle_category()

    def get_obj_position(self, name):
        obj_id = self.rom.get_object_by_handle(name).object_id
        index = self.sim.scene_obj_ids.index(obj_id)
        pos = self.scene_pos[index]
        return pos

    def get_objects(self):
        self.objects = {}
        for name in self.name_to_receptacle.keys():
            position = self.get_obj_position(name)
            obj = Object(name, self.name_to_receptacle[name], position)
            self.objects[name] = obj

    def group_objects_by_receptacle_category(self):
        self.grouped_objects = {}
        for name, obj in self.objects.items():
            if obj.receptacle_category not in self.grouped_objects:
                self.grouped_objects[obj.receptacle_category] = []
            self.grouped_objects[obj.receptacle_category].append(obj)

    def get_accesible_objects(self):
        return [obj for name, obj in self.objects.items() if obj.inside_receptacle]

    def print_accesible_objects(self):
        for obj in self.get_accesible_objects():
            print(obj.name, obj.pos)

    def set_dynamic_target(self, target):
        obj = self.objects[target]
        self.sim.dynamic_target = obj.pos
        print(f"The target is set to {obj.short_name} in the {obj.receptacle_category}")
