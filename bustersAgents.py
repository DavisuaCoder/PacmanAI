from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    past_positions = []
    num_living_ghosts = 999

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        
        distance_to_target = 999
        target = 0
        num_ghosts = 0
        reset = False
        
        # CALCULAR RESETEO EN CASILLAS PASADAS
        for i in range(0, gameState.getNumAgents() - 1):
            if gameState.getLivingGhosts()[i+1] == True:
                num_ghosts = num_ghosts +1
                for j in range(len(BasicAgentAA.past_positions)):
                    '''PROBAR TUPLA ENTERA'''
                    if gameState.getGhostPositions()[i]==BasicAgentAA.past_positions[j]:
                        reset = True

        if len(BasicAgentAA.past_positions) != 0:
            if reset == True or num_ghosts < BasicAgentAA.num_living_ghosts:
                BasicAgentAA.num_living_ghosts = num_ghosts
                BasicAgentAA.past_positions.clear()
               
        # ESTABLECER MOVIMIENTO LEGALES
        for i in range(len(BasicAgentAA.past_positions)):
            if gameState.getPacmanPosition()[0] - BasicAgentAA.past_positions[i][0] == 1 and gameState.getPacmanPosition()[1] - BasicAgentAA.past_positions[i][1] == 0:
                legal.remove(Directions.WEST)
            elif gameState.getPacmanPosition()[0] - BasicAgentAA.past_positions[i][0] == -1 and gameState.getPacmanPosition()[1] - BasicAgentAA.past_positions[i][1] == 0:
                legal.remove(Directions.EAST)
            elif gameState.getPacmanPosition()[0] - BasicAgentAA.past_positions[i][0] == 0 and gameState.getPacmanPosition()[1] - BasicAgentAA.past_positions[i][1] == 1:
                legal.remove(Directions.SOUTH)
            elif gameState.getPacmanPosition()[0] - BasicAgentAA.past_positions[i][0] == 0 and gameState.getPacmanPosition()[1] - BasicAgentAA.past_positions[i][1] == -1:
                legal.remove(Directions.NORTH) 
        
       # EVITAR QUE PACMAN SE ENCIERRE
        if len(legal) == 1 and legal[0]=='Stop':
            legal = gameState.getLegalActions(0)
            BasicAgentAA.past_positions.clear()

        # MOVIMIENTO
        # ELEGIR OBJETIVO
        for i in range(0, gameState.getNumAgents() - 1):
            if gameState.getLivingGhosts()[i+1] == True:
                if distance_to_target > gameState.data.ghostDistances[i]:
                    distance_to_target= gameState.data.ghostDistances[i]
                    target = i

        # MOVER HACIA EL OBJETIVO PRIMERO HORIZONTAL LUEGO VERTICAL 
        # SI NO ES POSIBLE ELEGIR DIRECCION LEGAL DE FORMA ALEATORIA
        if gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[target][0] > 0 and Directions.WEST in legal:
            move = Directions.WEST
        elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[target][0] < 0 and Directions.EAST in legal:
            move = Directions.EAST
        elif gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[target][1] > 0 and Directions.SOUTH in legal:
            move = Directions.SOUTH
        elif gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[target][1] < 0 and Directions.NORTH in legal:
            move = Directions.NORTH
        else:
            while(move == Directions.STOP):
                move_random = random.randint(0, 3)
                if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
                elif   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
                elif   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
                elif   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        
        BasicAgentAA.past_positions.append(gameState.getPacmanPosition())
        return move

    def printLineData(self, gameState):
        import numpy as np
        output = []
        info = []
        info.append(self.countActions)#ticks
        info.extend(list(gameState.getPacmanPosition()))
        info.append(gameState.data.agentStates[0].getDirection())
        livingGhosts = gameState.getLivingGhosts().copy()
        livingGhosts.pop(0)
        info.extend(livingGhosts)
        info.extend(gameState.data.ghostDistances)
        info.append(gameState.getNumFood())
        info.append(gameState.getDistanceNearestFood())
        output.append(info)

        with open('pacman-info.txt', 'a') as file:
            np.savetxt(file, output, delimiter=",", fmt="%s")

class QLearningAgent(BustersAgent):

    def initValues(self):
        "Initialize Q-values"
        self.actions = {Directions.NORTH:0, Directions.EAST:1, Directions.SOUTH:2, Directions.WEST:3}
        self.createQtable(256, "qTable.txt")
        self.table_file = open("qTable.txt", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.0
        self.alpha = 0.0
        self.discount = 0.8
        self.target = 0
        self.dicRow = ["[0],1","[0],2","[0],3","[1],0","[1],2","[1],3","[2],0","[2],1","[2],3","[3],0","[3],1","[3],2",
                       "[0, 1],2","[0, 1],3","[0, 2],1","[0, 2],3","[0, 3],1","[0, 3],2","[1, 2],0","[1, 2],3","[1, 3],0","[1, 3],2","[2, 3],0","[2, 3],1",
                       "[0, 1, 2],3","[0, 1, 3],2","[0, 2, 3],1","[1, 2, 3],0",
                       "0","1","2","3"]
    
    def registerInitialState(self, gameState):
        self.initValues()
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
    
    def createQtable(self, rows, nombreFichero):
        import numpy as np
        import os
        if not os.path.exists(nombreFichero):
            qTable = []
            emptyRow = [0.0, 0.0, 0.0, 0.0]
            for i in range(rows):
                qTable.append(emptyRow)
            
            with open(nombreFichero, 'a') as file:
                np.savetxt(file, qTable, delimiter=" ", fmt="%s")

    def printQtable(self):
        i = 0
        "Print qtable"
        for line in self.q_table:
            print(i, line)
            i +=1
        print("\n")     

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        print("------------TICK : ",self.countActions)
        move = Directions.STOP
        quadrant = self.getQuadrant(gameState)#Obtenemos el quadrante del objetivo
        print("quadrant ", quadrant-1)
        move = self.getActionQValues(quadrant, gameState)
        print("action ", move)
        return move
    
    def getQuadrant(self, gameState):
        self.target = self.getTarget(gameState)
        quadrant = 0
        #OBJETIVO FANTASMA
        if not self.target == 4:
            if gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] > 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] > 0:
                quadrant = 3
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] < 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] > 0:
                quadrant = 4
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] < 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] < 0:
                quadrant = 1
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] > 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] < 0:
                quadrant = 2
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] == 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] < 0:
                quadrant = 5 #Arriba
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] == 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] > 0:
                quadrant = 6 #Abajo
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] > 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] == 0:
                quadrant = 7 #izquierda
            elif gameState.getPacmanPosition()[0] - gameState.getGhostPositions()[self.target][0] < 0 and gameState.getPacmanPosition()[1] - gameState.getGhostPositions()[self.target][1] == 0:
                quadrant = 8 #derecha
        #OBJETIVO COMIDA
        else:
            if gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] > 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] > 0:
                quadrant = 3
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] < 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] > 0:
                quadrant = 4
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] < 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] < 0:
                quadrant = 1
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] > 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] < 0:
                quadrant = 2
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] == 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] < 0:
                quadrant = 5
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] == 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] > 0:
                quadrant = 6
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] > 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] == 0:
                quadrant = 7
            elif gameState.getPacmanPosition()[0] - gameState.getNearestFoodPosition()[0] < 0 and gameState.getPacmanPosition()[1] - gameState.getNearestFoodPosition()[1] == 0:
                quadrant = 8
        return quadrant
    
    def getTarget(self, gameState):
        computeTarget = 0 
        distance_to_target = 999
        # OBJETIVO MÁS CERCANO
        if self.checkLastGhost(gameState) == 1 and gameState.getDistanceNearestFood() != None:#PRIORIZAMOS LA COMIDA SI QUEDA 1 FANTASMA
            computeTarget = 4
        else:
            for i in range(0, gameState.getNumAgents() - 1): #0,1,2,3
                if gameState.getLivingGhosts()[i+1] == True:
                    if distance_to_target > gameState.data.ghostDistances[i]:
                        distance_to_target= gameState.data.ghostDistances[i]
                        computeTarget = i
            

            if (gameState.getDistanceNearestFood() != None and distance_to_target > gameState.getDistanceNearestFood()):
                computeTarget = 4
        
        return computeTarget

    def checkLastGhost(self, gameState):
        lastGhost = 0

        for i in range(0, gameState.getNumAgents() - 1): #0,1,2,3
            if gameState.getLivingGhosts()[i+1] == True:
                lastGhost += 1
        
        return lastGhost
    
    def computePosition(self, state, gameState):
        """
        Compute the row of the qtable for a given state.
        """
        wallDirections = [0,1,2,3]#POSIBLES MUROS 0:N - 1:E - 2:S - 3:W
        targetPosition = [0,0]
        key = ""
        exitWall = ""
        row = (state-1)*32
        legalActions = gameState.getLegalPacmanActions()
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        for action in legalActions:
            wallDirections.remove(self.actions[action])
        

        #OBTENEMOS LA POSICION DEL TARGET
        if self.target == 4:
            targetPosition = gameState.getNearestFoodPosition()
        else:
            targetPosition = gameState.getGhostPositions()[self.target]

        exitWall = gameState.data.layout.possiblePath(gameState.getPacmanPosition(),targetPosition)

        if len(wallDirections) == 0: #SIN MURO
            key = exitWall
        else:
            key = str(wallDirections)+","+exitWall
        
        return row + self.dicRow.index(key)

    def getQValue(self, state, action, gameSate):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state, gameSate)
        action_column = self.actions[action]

        return self.q_table[position][action_column]


    def computeValueFromQValues(self, state, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = gameState.getLegalPacmanActions()
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state, gameState)])

    def computeActionFromQValues(self, state, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = gameState.getLegalPacmanActions()
        if "Stop" in legalActions:
            legalActions.remove("Stop")

        if len(legalActions)==0:
          return None
    
        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0], gameState)
        for action in legalActions:
            value = self.getQValue(state, action, gameState)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getActionQValues(self, state,  gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = gameState.getLegalPacmanActions()
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
             return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state, gameState)


    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        if state.getDistanceNearestFood() != None and reward == 199:
            print("Me he comido al fantasma y todavía queda comida")
            reward = -1

        print("Update Q-table with transition: ", state, action, nextState, reward)
        print("quadrant ",self.getQuadrant(state)-1)
        print("action ", action)
        position = self.computePosition(self.getQuadrant(state), state)
        action_column = self.actions[action]
        print("Corresponding Q-table cell to update:", position, action_column)

        
        "*** YOUR CODE HERE ***"
        self.q_table[position][action_column]= (1-self.alpha)*self.getQValue(self.getQuadrant(state), action, state) + self.alpha*((reward+1) + self.discount * self.computeValueFromQValues(self.getQuadrant(nextState), nextState))
        



        #TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        print("")
        print("----------------------------------------------------------------------------------------------------------------")
        # print("Q-table:")
        # self.printQtable()

    def getPolicy(self, state, gameState):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state, gameState)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

