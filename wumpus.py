import numpy as np
from PIL import Image
from PIL import ImageDraw
import sys
import os

class Orientation:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class Actions:
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5

class Percepts:
    # [stench,breeze,glitter,bump,scream]
    STENCH = 0
    BREEZE = 1
    GLITTER = 2
    BUMP = 3
    SCREAM = 4

def get_concat_h(im1, im2):
    """
    Concatenate two images
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class WumpusWorld:
    def __init__(self, seed=2026, size=(4,4), p_pit=0.2, Tmax=100):
        """
        seed ... seed for random number generator
        size ... size of the grid of the Wumpus world
        p_pit ... probability of a pit
        """
        self.rng = np.random.default_rng(seed=seed)
        self.size = size
        self.p_pit = p_pit

        ## for drawing
        self.wall_width = 3 # wall width in pixels
        self.cell_size = 100 # cell size in pixels for drawing

        self.terminated = True
        self.Tmax = Tmax

    def __load_icons(self):
        """
        load icons
        """
        self.icons = {}

        # [stench,breeze,glitter,bump,scream]

        for name, filename in [("wumpus", "monster.png"), ("gold", "gold.png"), ("pit", "pit.png"), ("agent", "robot.png"), ("north", "robot-north.png"), ("east", "robot-east.png"), ("south", "robot-south.png"), ("west", "robot-west.png"), ("stench", "stench.png"), ("breeze", "breeze.png"), ("glitter", "glitter.png"), ("bump", "bump.png"), ("scream", "scream.png")]:
            icon = Image.open(os.path.join("icons", filename))
            icon = icon.resize((30,30), Image.Resampling.LANCZOS)
            self.icons[name] = icon
            
        self._convert_icons_to_pygame()

    def __index_to_pos(self, index):
        """
        Convert index to (x,y)-coordinates
        """
        y = index // self.size[0]
        x = index - y * self.size[0]

        return (x,y)

    def reset(self):
        """
        Generate a new Wumpus world
        """
        self.pos_agent = (0,0)
        self.orientation_agent = Orientation.NORTH
        self.has_arrow = True
        self.has_gold = False
        self.terminated = False
        self.wumpus_alive = True
        self.reward = 0

        # generate pits
        pits = self.rng.random(self.size) <= self.p_pit
        pits[0,0] = False
        self.pits = np.hstack([x.reshape((-1,1))for x in np.where(pits)])

        # select position for wumpus
        idx_wumpus = self.rng.integers(low=1, high=np.prod(self.size))
        self.pos_wumpus = self.__index_to_pos(idx_wumpus)

        # select position for gold
        idx_gold = self.rng.integers(low=0, high=np.prod(self.size))
        self.pos_gold = self.__index_to_pos(idx_gold)

        # generate initial percept
        obs = obs = np.zeros((5,), dtype=int)

        stench = self.__is_stench()
        breeze = self.__is_breeze()
        glitter = self.__is_glitter()
        bump = False
        scream = False
        
        if stench:
            obs[Percepts.STENCH] = True
        if breeze:
            obs[Percepts.BREEZE] = True
        if glitter:
            obs[Percepts.GLITTER] = True
        if bump:
            obs[Percepts.BUMP] = True
        if scream:
            obs[Percepts.SCREAM] = True

        self.obs = obs
        self.t = 0

        return self.obs

    def __offset_for_pos(self, pos):
        """
        Get the offset for drawing for a position (top left of a cell)
        """
        x_offset = self.wall_width + pos[0] * (self.cell_size + self.wall_width)
        y_offset = self.wall_width + self.size[1] * (self.wall_width + self.cell_size) - ((1 + pos[1]) * (self.cell_size + self.wall_width))

        return (x_offset, y_offset)
    
    def _convert_icons_to_pygame(self):
        """
        Converts all PIL images in self.icons to pygame Surfaces.
        Handles transparency (RGBA) automatically.
        """
        for key, pil_img in self.icons.items():
            # Ensure the image is in RGBA to preserve transparency
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            
            data = pil_img.tobytes()
            size = pil_img.size
            
            # Create pygame surface from string
            pygame_surf = pygame.image.fromstring(data, size, 'RGBA')
            
            # Optimize for pygame performance
            self.icons[key] = pygame_surf.convert_alpha()

    def get_img_size(self):
        width = self.cell_size * self.size[0] + self.cell_size + 150
        height = self.cell_size * self.size[1] + self.wall_width * (self.size[1] + 1)

        return (width, height)

    def render(self):
        """
        Render the current state of the wumpus world using Pygame
        """
        if not hasattr(self, 'icons'):
            self.__load_icons()

        # 1. Calculate dimensions
        grid_h = self.size[1] * (self.cell_size + self.wall_width) + self.wall_width
        grid_w = self.size[0] * (self.cell_size + self.wall_width) + self.wall_width
        
        # Create the main grid surface
        grid_surf = pygame.Surface((grid_w, grid_h))
        grid_surf.fill((255, 255, 255)) # Background

        # 2. Draw Grid Lines
        sidx = 0
        for i in range(self.size[0] + 1):
            pygame.draw.rect(grid_surf, (0, 0, 0), (sidx, 0, self.wall_width, grid_h))
            sidx += self.wall_width + self.cell_size
        
        sidx = 0
        for i in range(self.size[1] + 1):
            pygame.draw.rect(grid_surf, (0, 0, 0), (0, sidx, grid_w, self.wall_width))
            sidx += self.wall_width + self.cell_size

        # 3. Draw Labels (Coordinates)
        # Note: Assumes self.font = pygame.font.SysFont('Sans', 10) is initialized in __init__
        font = pygame.font.SysFont('Sans', 12)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                x_pos = self.wall_width + 3 + x * (self.cell_size + self.wall_width)
                # Pygame's Y-axis starts from top, so we flip the Y coordinate logic if needed
                y_pos = grid_h - self.wall_width - 15 - y * (self.cell_size + self.wall_width)
                text_surf = font.render(f"({x+1},{y+1})", True, (0, 0, 0))
                grid_surf.blit(text_surf, (x_pos, y_pos))

        # 4. Helper for offsets (Update this method to return screen coordinates)
        def get_pygame_offset(pos):
            # pos is (x, y)
            ox = pos[0] * (self.cell_size + self.wall_width) + self.wall_width
            oy = (self.size[1] - 1 - pos[1]) * (self.cell_size + self.wall_width) + self.wall_width
            return (ox, oy)

        # 5. Show Icons (Wumpus, Gold, Pits, Agent)
        # Note: This assumes self.icons contains pygame.Surface objects, not PIL Images
        if self.wumpus_alive:
            offset = get_pygame_offset(self.pos_wumpus)
            grid_surf.blit(self.icons["wumpus"], (offset[0] + 10, offset[1] + 10))

        if self.pos_gold is not None:
            offset = get_pygame_offset(self.pos_gold)
            grid_surf.blit(self.icons["gold"], (offset[0] + 10 + self.cell_size // 2, offset[1] + 10))

        for pit in self.pits:
            offset = get_pygame_offset(pit)
            grid_surf.blit(self.icons["pit"], (offset[0] + 10, offset[1] + 5 + self.cell_size // 2))

        # Agent
        offset = get_pygame_offset(self.pos_agent)
        orient_map = {
            Orientation.NORTH: "north", Orientation.EAST: "east",
            Orientation.SOUTH: "south", Orientation.WEST: "west"
        }
        agent_icon = self.icons[orient_map[self.orientation_agent]]
        grid_surf.blit(agent_icon, (offset[0] + 10 + self.cell_size // 2, offset[1] + 5 + self.cell_size // 2))

        # 6. Percepts Pane
        percept_surf = pygame.Surface((self.cell_size, grid_h))
        percept_surf.fill((240, 240, 240))
        percept_surf.blit(font.render("Percepts", True, (0, 0, 0)), (10, 10))
        
        percept_list = [(Percepts.STENCH, "stench"), (Percepts.BREEZE, "breeze"), 
                        (Percepts.GLITTER, "glitter"), (Percepts.BUMP, "bump"), (Percepts.SCREAM, "scream")]
        for i, k in percept_list:
            if self.obs[i]:
                percept_surf.blit(self.icons[k], (10, 35 + i * 45))

        # 7. Status Pane
        status_surf = pygame.Surface((150, grid_h))
        status_surf.fill((255, 255, 255))
        lines = [
            f"Reward: {self.reward}",
            f"Terminated: {self.terminated}",
            f"Arrow: {self.has_arrow}",
            f"Gold: {self.has_gold}"
        ]
        for i, line in enumerate(lines):
            status_surf.blit(font.render(line, True, (0, 0, 0)), (10, 10 + (i * 20)))

        # 8. Combine Surfaces (Horizontal Concat)
        total_width = grid_w + percept_surf.get_width() + status_surf.get_width()
        final_surf = pygame.Surface((total_width, grid_h))
        final_surf.blit(grid_surf, (0, 0))
        final_surf.blit(percept_surf, (grid_w, 0))
        final_surf.blit(status_surf, (grid_w + percept_surf.get_width(), 0))

        return final_surf
    
    def __is_glitter(self):
        # are we on a field with gold
        if self.pos_agent == self.pos_gold:
            return True
        else:
            return False
        
    def __is_stench(self):
        # are we next to a wumpus?
        if self.pos_wumpus is None:
            return False
        dx = self.pos_agent[0] - self.pos_wumpus[0]
        dy = self.pos_agent[1] - self.pos_wumpus[1]
        if np.abs(dx) + np.abs(dy) <= 1:
            return True
        else:
            return False
        
    def __is_breeze(self):
        # are we next to a pit?
        for pos_pit in self.pits:
            dx = self.pos_agent[0] - pos_pit[0]
            dy = self.pos_agent[1] - pos_pit[1]
            if np.abs(dx) + np.abs(dy) == 1:
                return True
        return False
    
    def step(self, action):
        if self.terminated:
            raise AssertionError("Environment already terminted. Reset before taking any further actions.")

        # percept variables
        # [stench,breeze,glitter,bump,scream]
        stench = False
        breeze = False
        glitter = False
        bump = False
        scream = False

        reward = -1 # any actions costs
        terminated = False
        info = {}
        if action == Actions.LEFT:
            self.orientation_agent = (self.orientation_agent - 1) % 4
        elif action == Actions.RIGHT:
            self.orientation_agent = (self.orientation_agent + 1) % 4
        elif action == Actions.FORWARD:
            old_pos = self.pos_agent
            dx = 0
            dy = 0
            if self.orientation_agent == Orientation.EAST:
                dx = 1
            elif self.orientation_agent == Orientation.WEST:
                dx = -1
            elif self.orientation_agent == Orientation.NORTH:
                dy = 1
            elif self.orientation_agent == Orientation.SOUTH:
                dy = -1

            new_x = max(0, min(self.size[0] - 1, self.pos_agent[0] + dx))
            new_y = max(0, min(self.size[1] - 1, self.pos_agent[1] + dy))
    
            new_pos = (new_x, new_y)
            if old_pos == new_pos:
                bump = True
            self.pos_agent = new_pos
        elif action == Actions.SHOOT:
            if self.has_arrow:
                reward -= 10
                self.has_arrow = False

                # do we hit the wumpus?
                if self.orientation_agent == Orientation.NORTH:
                    dx = 0
                    dy = 1
                elif self.orientation_agent == Orientation.EAST:
                    dx = 1
                    dy = 0
                elif self.orientation_agent == Orientation.SOUTH:
                    dx = 0
                    dy = -1
                elif self.orientation_agent == Orientation.WEST:
                    dx = -1
                    dy = 0
                for i in range(max(self.size)):
                    pos_arrow = (self.pos_agent[0] + i*dx, self.pos_agent[1] + i*dy)
                    if self.pos_wumpus == pos_arrow:
                        self.wumpus_alive = False
                        scream = True
        elif action == Actions.GRAB:
            if self.pos_agent == self.pos_gold:
                self.has_gold = True
                self.pos_gold = None
        elif action == Actions.CLIMB:
            if self.pos_agent == (0,0): 
                terminated = True
                if self.has_gold:
                    reward += 1000

        # compile percepts, check for death
        if self.wumpus_alive and (self.pos_agent == self.pos_wumpus):
            reward -= 1000
            terminated = True
        for pos_pit in self.pits:
            pos_pit = (pos_pit[0], pos_pit[1])
            if self.pos_agent == pos_pit:
                reward = -1000
                terminated = True

        if self.pos_gold == self.pos_agent:
            glitter = True

        # compile observation vector
        #   [stench,breeze,glitter,bump,scream]
        obs = np.zeros((5,), dtype=int)
        stench = self.__is_stench()
        breeze = self.__is_breeze()
        glitter = self.__is_glitter()
        if stench:
            obs[Percepts.STENCH] = True
        if breeze:
            obs[Percepts.BREEZE] = True
        if glitter:
            obs[Percepts.GLITTER] = True
        if bump:
            obs[Percepts.BUMP] = True
        if scream:
            obs[Percepts.SCREAM] = True

        self.obs = obs

        self.t += 1
        if not terminated:
            if self.t >= (self.Tmax - 1):
                reward -= 1000
                terminated = True
                info = "Max steps reached."

        self.terminated = terminated
        self.reward = reward
        return obs, reward, terminated, info




if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    import pygame
    pygame.init()
    pygame.display.set_mode()

    # Initialize the wumpus world
    wumpus_world = WumpusWorld(1234, size=(4,4))
    display = pygame.display.set_mode(wumpus_world.get_img_size())
    wumpus_world.reset()
    
    # render
    surf = wumpus_world.render()
    display.blit(surf, (0, 0))
    pygame.display.update()

    # creating a running loop
    pygame.event.clear()
    while True:
        # creating a loop to check events that
        # are occurring
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                wumpus_world.step(Actions.FORWARD)
            elif event.key == pygame.K_LEFT:
                wumpus_world.step(Actions.LEFT)
            elif event.key == pygame.K_RIGHT:
                wumpus_world.step(Actions.RIGHT)
            elif event.key == pygame.K_g:
                wumpus_world.step(Actions.GRAB)
            elif event.key == pygame.K_c:
                wumpus_world.step(Actions.CLIMB)
            elif event.key == pygame.K_t:
                wumpus_world.step(Actions.SHOOT)
            elif event.key == pygame.K_r:
                wumpus_world.reset()
            elif event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

            surf = wumpus_world.render()
            display.blit(surf, (0, 0))
            pygame.display.update()
