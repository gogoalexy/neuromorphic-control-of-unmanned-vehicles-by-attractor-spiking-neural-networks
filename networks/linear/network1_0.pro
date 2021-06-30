DefineMacro

GroupName:Ordinal
GroupMembers:Ord0,Ord1,Ord2,Ord3,Ord4
EndGroupMembers

GroupName:Task
GroupMembers:Task0,Task1,Task2,Task3,Task4
EndGroupMembers

GroupName:Shifter
GroupMembers:S0,S1,S2,S3,S4
EndGroupMembers

EndDefineMacro


EventTime 500.0
Type=ChangeExtFreq
Label=#1#
Population:Ord0
Receptor:AMPA
FreqExt=400
EndEvent

EventTime 550.0
Type=ChangeExtFreq
Label=#1#
Population:Ord0
Receptor:AMPA
FreqExt=0
EndEvent

EventTime 1000.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=260
EndEvent

EventTime 1050.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=0
EndEvent

EventTime 1500.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=260
EndEvent

EventTime 1550.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=0
EndEvent

EventTime 2000.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=260
EndEvent

EventTime 2050.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=0
EndEvent

EventTime 2500.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=260
EndEvent

EventTime 2550.0
Type=ChangeExtFreq
Label=#1#
Population:Next
Receptor:AMPA
FreqExt=0
EndEvent

EventTime 3000.00
Type=EndTrial
Label=End_of_the_trial
EndEvent

OutControl
FileName:ordinal.dat
Type=Spike
population:Ordinal
EndOutputFile

FileName:task.dat
Type=Spike
population:Task
EndOutputFile

FileName:shifter.dat
Type=FiringRate
FiringRateWinodw=50
PrintStep=10
population:shifter
EndOutputFile

%FileName:inhibition.dat
%Type=FiringRate
%FiringRateWinodw=50
%PrintStep=10
%population:Inh
%EndOutputFile

EndOutControl
