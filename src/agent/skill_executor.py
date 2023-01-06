from agent.motor_skills import NavSkillPolicy, PickSkillPolicy, PlaceSkillPolicy, ResetArmSkill

class SkillExecutor():
    def __init__(self, Env, device, skill_id):
        self.env = Env
        self.conf = Env.conf
        self.skill_id = skill_id
        self.device = device
        self.skill = self.__init_skill(skill_id)
        self.reset = self.__init_skill('reset_arm')

    def __init_skill(self, skill_id):
        skill_config = self.conf.agent.motor_skills[skill_id].rl
        cls = eval(skill_config.skill_name)
        skill = cls.from_config(
            skill_config,
            self.env.internal_observation_space,
            self.env.action_space,
            1,
            self.conf
        )
        skill.to(self.device)
        return skill
    
    def get_gripper_status(self):
        snapped = self.env.sim.grasp_mgr.snap_idx
        if snapped:
            return f'Holding {self.env.scene_parser.rom.get_object_by_id(snapped).handle}'
        else:
            return None

    def run(self, target):
        self.env.execute_low_level_skill(self.reset)
        try:
            self.env.set_target(target)
        except KeyError:
            return "Failure, send one valid id. Remember to use the find functions before"

        finished, failed = self.env.execute_low_level_skill(self.skill)
        status = self.get_gripper_status()

        if finished:
            return f'Successfull execution - {status}'
        elif failed:
            return 'Unexpected failure'

    def from_config(env, device, skill_id):
        return SkillExecutor(env, device, skill_id)

