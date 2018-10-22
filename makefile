# Cursor commands
# cliclick -e 1000 -m verbose c:0,450
cursor-center:
	cliclick -e 1000 -m verbose c:$$(make center-x),$$(make center-y)

cursor-top:
	cliclick -e 1000 -m verbose c:$$(make center-x),$$(make top-y)

cursor-down:
	cliclick -e 1000 -m verbose c:$$(make center-x),$$(make down-y)

cursor-left:
	cliclick -e 1000 -m verbose c:$$(make left-x),$$(make center-y)

cursor-right:
	cliclick -e 1000 -m verbose c:$$(make right-x),$$(make center-y)

# Hardcoded values =/
center-x:
	@echo 720

center-y:
	@echo 450

top-y:
	@echo 0

down-y:
	@echo "$$(($$(make max-height)-10))"

left-x:
	@echo 0

right-x:
	@echo "$$(($$(make center-x)*2))"

# Get desktop bounds in real pixels via AppleScript
desktop-bounds:
	@osascript -e 'tell application "Finder" to get bounds of window of desktop'

min-width:
	@osascript -e 'tell application "Finder" to set {MinWidth} to {item 1} of (get bounds of window of desktop)'

min-height:
	@osascript -e 'tell application "Finder" to set {MinHeight} to {item 2} of (get bounds of window of desktop)'

max-width:
	@osascript -e 'tell application "Finder" to set {MaxWidth} to {item 3} of (get bounds of window of desktop)'

max-height:
	@osascript -e 'tell application "Finder" to set {MaxHeight} to {item 4} of (get bounds of window of desktop)'

# Information taken from screen (Hardware)
display-info:
	@system_profiler SPDisplaysDataType

display-resolution:
	@make display-info | awk '/Resolution/{print $$2, $$4}'

display-width:
	@make display-info | awk '/Resolution/{print $$2}'

display-height:
	@make display-info | awk '/Resolution/{print $$4}'

display-y:
	@make display-height | while read height; do echo "$$(($$height/2))"; done;

display-x:
	@make display-width | while read width; do echo "$$(($$width/2))"; done;