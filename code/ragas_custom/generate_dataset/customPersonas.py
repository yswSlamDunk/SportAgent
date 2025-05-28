from ragas.testset.persona import Persona

persona_general_fitness_beginner = Persona(
    name="General Fitness Beginner",
    role_description="A non-athletic person starting weightlifting for general health and fitness improvement. Needs simple explanations and safety assurance, interested in basic exercise methods and precautions.",
)

persona_diet_motivated_beginner = Persona(
    name="Diet-Motivated Beginner",
    role_description="Primary goal is body transformation and weight loss, interested in weightlifting-based routines and diet plans. Values information about exercise effectiveness and calorie consumption.",
)

persona_home_training_beginner = Persona(
    name="Home Training Beginner",
    role_description="User who wants to start weightlifting at home without going to the gym. Looking for alternative exercises with minimal equipment and workout methods within space constraints.",
)

persona_middle_aged_beginner = Persona(
    name="Middle-Aged Beginner",
    role_description="User aged 40-60+ who is sensitive to injury prevention and posture correction. Pursues safe exercise methods considering joint health and overall physical condition.",
)

persona_fitness_gym_novice = Persona(
    name="Fitness Gym Novice",
    role_description="Beginner who has received basic training at the gym but is new to weightlifting. Hopes for proper feedback on form and movement.",
)

persona_youth_athlete_aspirant = Persona(
    name="Youth Weightlifting Aspirant",
    role_description="Teenager preparing for physical education-related college admission, interested in natural connection movements.",
)

persona_amateur_competitor = Persona(
    name="Amateur Weightlifting Competitor",
    role_description="Preparing for amateur weightlifting competitions, interested in weight management and pre-competition routine planning. Needs information about competition preparation process and weight management.",
)

persona_performance_optimizer = Persona(
    name="Weightlifting Performance Optimizer",
    role_description="Pursues technical perfection rather than just record improvement in weightlifting, seeks biomechanical-based feedback. Demands detailed analysis of accurate posture and efficient movement.",
)

personas = [persona_general_fitness_beginner,
    persona_diet_motivated_beginner,
    persona_home_training_beginner,
    persona_middle_aged_beginner,
    persona_fitness_gym_novice,
    persona_youth_athlete_aspirant,
    persona_amateur_competitor,
    persona_performance_optimizer]