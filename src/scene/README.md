# Scene Parser

The scene parser is a tool to parse the scene from the simulator. It is used as a mechanism
to get objects and receptacles currently on the scene directly from the position as well
as their positions.


## Usage

Instantiate 

```
from scene.scene_parser import SceneParser
scene = SceneParser(env.sim)
```

### Objects:
```
scene.objects.keys()
>>> dict_keys(['021_bleach_cleanser_:0000', '012_strawberry_:0000', ...])

scene.get_obj_position('005_tomato_soup_can_:0002')
>>> array([0.55986, 0.7589 , 6.53018]
```

### Receptacles
```
scene.receptacles.keys()
>>> dict_keys(['frl_apartment_chair_01_:0000', 'frl_apartment_chair_01_:0001', ...])
```


Receptacles can be articulated or rigid an can have from 1 to several surfaces
```
scene.receptacles['fridge_:0000']
>>> Fridge [fridge_:0000] with 5 surfaces and is of type articulated
```

We can get the center location of a receptacle by running:
```
scene.get_receptacle_position('fridge_:0000')
>>> 
```

The center of a receptacle should be good for navigation but for more dexterous manipulation like placing, we need to know a position in one of the surfaces, for example we can sample a 
random position in the surface of the fridge:
```
scene.receptacles['fridge_:0000'].surfaces[0].name
>>> receptacle_aabb_bottomfrl_apartment_refrigerator

scene.sample_position_in_receptacle('fridge_:0000','receptacle_aabb_...')
```





