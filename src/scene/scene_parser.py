from habitat.datasets.rearrange.samplers.receptacle import parse_receptacles_from_user_config

receptacle_categories = {
    "Chair": ["Chr", 'chair'],
    "Sofa": ["Sofa", 'sofa'],
    "Table": ["Tbl", 'table'],
    "Drawer": ["drawer"],
    "Sink": ["sink"],
    "Tv Stand": ["TvStnd", 'tvstand'],
    "Counter": ["counter"],
    "Fridge": ["fridge", 'refrigerator'],
    'Wall Cabinet': ['wall_cabinet'],
    'Kitchen Cupboard': ['kitchenCupboard']
}


def get_receptacle_category(receptacle_name):
    for k, values in receptacle_categories.items():
        for value in values:
            if value in receptacle_name:
                return k
    print('Failed', receptacle_name)

class Receptacle:
    def __init__(self, name, receptacle, surfaces, type):
        self.name = name
        self.receptacle = receptacle
        self.surfaces = surfaces
        self.pos = self.receptacle.translation
        self.type = type
        self.receptacle_category = get_receptacle_category(self.surfaces[0].name)
    
    def __repr__(self) -> str:
        return f"{self.receptacle_category} [{self.name}] with {len(self.surfaces)} surfaces and is of type {self.type}"

class Object:
    def __init__(self, name, receptacle, pos):
        self.name = name
        self.receptacle = receptacle
        self.pos = pos

        self.receptacle_category = get_receptacle_category(self.receptacle)
        self.short_name = " ".join(self.name.split("_")[1:-1])
        #self.inside_receptacle = self.get_accesibility()

    def get_short_name(self):
        self.short_name = " ".join(self.name.split("_")[1:-1])

    def get_accesibility(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.short_name} [{self.name} in {self.receptacle} {self.receptacle_category}]"


class SceneParser:
    def __init__(self, sim):   
        # Set arguments
        self.sim = sim
        self.rom = self.sim.get_rigid_object_manager()
        self.aom = self.sim.get_articulated_object_manager()
        
        # Get all objects and receptacles in the scene
        self.name_to_receptacle = self.sim.ep_info["name_to_receptacle"]

        # Get receptacle objects in the scene
        self.receptacles = self.extract_receptacles()
        self.receptacles = {r.name: r for r in self.receptacles}

        self.surfaces = {}
        for r in self.receptacles.values():
            for surface in r.surfaces:
                self.surfaces[surface.name] = surface

        # Get object positions in the scene
        self.scene_pos = self.sim.get_scene_pos()

        # Get parsed objects
        self.objects = self.get_objects()

        # Get objects and receptacles grouped by an interpretable category
        self.group_objects_by_receptacle_category()
        self.group_receptacles_by_category()

    def get_objects(self):
        objects = {}
        for name in self.name_to_receptacle.keys():
            position = self.get_obj_position(name)
            obj = Object(name, self.name_to_receptacle[name], position)
            objects[name] = obj
        return objects

    def get_obj_position(self, name):
        '''
            The object position is extracted of the scene because the scene
            can be modified and the object will move, right now this is 
            fixed but self.scene_pos can be updated
        '''
        obj_id = self.rom.get_object_by_handle(name).object_id
        index = self.sim.scene_obj_ids.index(obj_id)
        return self.scene_pos[index]
    
    def get_receptacle_position(self, receptacle_name):
        '''
            The receptacle position is an atribute of the receptacle because 
            its unchanged
        ''' 
        receptacle = self.receptacles[receptacle_name]
        return receptacle.pos

    def sample_position_in_surface(self, surface, sample_region_scale=1):
        try:
            surface = self.surfaces[surface]
        except KeyError:
            raise ValueError(f"{surface} does not exist")
            
        return surface.sample_uniform_global(self.sim, sample_region_scale)

    def set_dynamic_target(self, target):

        if target in self.objects:
            self.sim.dynamic_target = self.get_obj_position(target)
        
        elif target in self.receptacles:
            self.sim.dynamic_target = self.get_receptacle_position(target)
        
        elif target in self.surfaces:
            self.sim.dynamic_target = self.sample_position_in_surface(target)

        else:
            raise Exception(f'{target} is not a valid target')

    def group_objects_by_receptacle_category(self):
        self.grouped_objects = {}
        for name, obj in self.objects.items():
            if obj.receptacle_category not in self.grouped_objects:
                self.grouped_objects[obj.receptacle_category] = []
            self.grouped_objects[obj.receptacle_category].append(obj)
    
    def group_receptacles_by_category(self):
        self.grouped_receptacles = {}
        for name, receptacle in self.receptacles.items():
            if receptacle.receptacle_category not in self.grouped_receptacles:
                self.grouped_receptacles[receptacle.receptacle_category] = []
            self.grouped_receptacles[receptacle.receptacle_category].append(receptacle)

    def extract_receptacles(self):
        receptacles = []

        # rigid receptacles
        receptacles = []
        for obj_handle in self.rom.get_object_handles():
            receptacle = self.rom.get_object_by_handle(obj_handle)
            user_attr = receptacle.user_attributes
            samplers = parse_receptacles_from_user_config(
                user_attr, parent_object_handle=obj_handle
            )
            if samplers:
                receptacles.append(Receptacle(obj_handle, receptacle, samplers, 'rigid'))

        # articulated receptacles
        for obj_handle in self.aom.get_object_handles():
            receptacle = self.aom.get_object_by_handle(obj_handle)
            user_attr = receptacle.user_attributes
            samplers = parse_receptacles_from_user_config(
                    user_attr,
                    parent_object_handle=obj_handle,
                    valid_link_names=[
                        receptacle.get_link_name(link)
                        for link in range(-1, receptacle.num_links)
                    ],
                    ao_uniform_scaling=receptacle.global_scale,
                )
            if samplers:
                receptacles.append(Receptacle(obj_handle, receptacle, samplers, 'articulated'))    
        return receptacles