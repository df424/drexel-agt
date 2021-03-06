# drexel-agt
Team Project for Algorithmic Game Theory Class (Drexel University Spring 2019)

<h2>Introduction</h2>
The goal of this project is to explore multiagent interaction in the context of game theory.

<h2>Team Members:</h2>
<ol>
  <li>Joey Wilson</li>
  <li>Ross Griebenow</li>
  <li>David Flanagan</li>
</ol>

<h2>Environment Setup</h2>
Run the following commands to install the required packages.</br>
<code>sudo pip install numpy</code></br>
<code>sudo pip install matplotlib</code></br>
<code>sudo pip install argparse</code></br>
<code>sudo pip install scipy</code></br>

<h2>Running the games</h2>
All of the games are run from main.py.  To run a game pass the appropriate name as the first command line argument.  The available games and their commands are shown in the table below.</br>
<table>
  <tr><th>Game</th><th>Command Arg</th></tr>
  <tr><td>Rock-Paper-Scissors</td><td>rps</td></tr>
  <tr><td>Prisoner's Dilemma</td><td>prisoner</td></tr>
  <tr><td>Chicken</td><td>chicken</td></tr>
</table>
<h4>Example Command</h4>
<code>python .\main.py rps -i 10000 --random-start --rs-min 0 -N 50</code><br>
<p><b>NOTE: When using the multiplicitive weights (multi-w) optimizer the policy must be initialized with weights greater than 0 so set --rs-min to 0 or higher.</b></p>
<h2>Command Line Arguments</h2>
<table>
  <tr><th>Argument</th><th>Type</th><th>Default</th><th>Description</th></tr>
  <tr><td>-v, --verbose</td><td>bool</td><td>false</td><td>Enables verbose printing while simulation.  This degrades performance considerably.</td>
  <tr><td>-i, --iterations</td><td>int</td><td>10000</td><td>Sets the number of steps to run the simulation or the number of episodes to run if the game is episodic.</td></tr>
  <tr><td>-N</td><td>int</td><td>1</td><td>Average data over N runs of the simulation.</td></tr>
  <tr><td>--off-policy</td><td>bool</td><td>false</td><td>If set to true will run all agents with a balanced random policy.</td></tr>
  <tr><td>-l, --learn-rate</td><td>float</td><td>0.01</td><td>Set the learning rate used in policy optimization.</td></tr>
  <tr><td>--random-start</td><td>bool</td><td>true</td><td>Randomly initialize policies between the parameters given by --rs-max and --rs-min</td></tr>
  <tr><td>--rs-max</td><td>float</td><td>1.0</td><td>Upper bound to use during random initialization.</td></tr>
  <tr><td>--rs-min</td><td>float</td><td>0</td><td>Lower bound to use during random initialization.</td></tr>
  <tr><td>--optimizer</td><td>string</td><td>multi-w</td><td>Select the type of optimizer to use when updating policies. Options right now are 'multi-w' for multiplicitive weights and 'td' for temporal difference.</td></tr>
  <tr><td>--use-softmax</td><td>bool</td><td>false</td><td>Setting this to true will cause the agent\'s to use softmax for action selection.</td></tr>
</table>
