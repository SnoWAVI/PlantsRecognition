from ClassPlants import Plants
import warnings

warnings.filterwarnings("ignore")

while True == True:
    print('\n' + '=' * 50)
    print('What do you want to do?:')
    print('=' * 50)
    print('t - Train new model')
    print('r - Recognize image')
    print('rv - Recognize with examples')
    print('ev - Evaluate on examples')
    print('i - Incremental learning with diagnostics')
    print('q - Quit')
    print('=' * 50)
    command = input('> ')
    match command.split():
        case ['t']:
            plants = Plants(True)
        case ['r']:
            plants = Plants(False)
            plants.recognize_quick(False)
        case ['rv']:
            plants = Plants(False)
            plants.recognize_visual(False)
        case ['ev']:
            plants = Plants(False)
            plants.evaluate_on_both_datasets()
        case ['i']:
            plants = Plants(False)
            plants.debug_model_output()
            plants.incremental_learning_with_diagnostics()
        case ['q']:
            break
        case _:
            print('Wrong command')