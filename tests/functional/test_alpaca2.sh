# == Task Spec ==
# Task name: alpaca2
# Prompt file: None
# System prompt file: None

# == Sample Rekeyed Data ==

# Example 1:
#   Prompt: What are the names of some famous actors that started their careers on Broadway?
#   Baseline model: gpt4_1106_preview
#   Baseline model response: Several famous actors started their careers on Broadway before making it big in film and television. Here are a few notable examples:

# 1. Sarah Jessica Parker - Before she was Carrie Bradshaw on "Sex and the City," Sarah Jessica Parker was a Broadway star, having appeared in productions like "Annie" as a child.

# 2. Meryl Streep - Meryl Streep's early career included Broadway productions such as "Trelawny of the 'Wells'" and "A Memory of Two Mondays / 27 Wagons Full of Cotton."

# 3. Hugh Jackman - Hugh Jackman won a Tony Award for his role in "The Boy from Oz" and has been known for his stage work as well as his film career.

# 4. Sutton Foster - Known for her television role in "Younger," Sutton Foster is also a Broadway legend with leading roles in shows like "Thoroughly Modern Millie" and "Anything Goes."

# 5. Kristen Bell - Before she was the voice of Anna in "Frozen" or the star of "The Good Place," Kristen Bell appeared in Broadway's "The Adventures of Tom Sawyer" and "The Crucible."

# 6. Audra McDonald - Audra McDonald is a renowned Broadway actress with a record-breaking number of Tony Awards. She's starred in "Ragtime," "Carousel," "Master Class," and more.

# 7. Nathan Lane - Nathan Lane is a Broadway veteran known for his roles in "The Producers," "A Funny Thing Happened on the Way to the Forum," and "Angels in America."

# 8. Idina Menzel - Before "Frozen" and "Wicked" made her a household name, Idina Menzel started on Broadway in shows like "Rent" and "Hair."

# 9. Lin-Manuel Miranda - Before "Hamilton" and "In the Heights" became huge hits, Lin-Manuel Miranda was performing on Broadway, eventually becoming a celebrated writer and actor.

# 10. Lea Michele - Prior to her role on "Glee," Lea Michele was a young Broadway actress in shows like "Les Misérables," "Ragtime," and "Spring Awakening."

# These actors are just a few examples of the many performers who have transitioned from the Broadway stage to broader fame in the entertainment industry. Broadway often serves as a proving ground for talent, and many actors continue to return to the stage throughout their careers.
#   Dataset/category: helpful_base

# Example 2:
#   Prompt: How did US states get their names?
#   Baseline model: gpt4_1106_preview
#   Baseline model response: The names of U.S. states are derived from a variety of languages and sources, including indigenous American languages, European languages (primarily English, Spanish, and French), as well as other historical influences. Here is a brief overview of how some states got their names:

# 1. **Indigenous American Languages**: Many state names come from Native American words or names of tribes, often from the languages of the indigenous peoples who lived in or near those areas.
#    - **Alabama**: Possibly derived from the Choctaw language, meaning "thicket clearers."
#    - **Connecticut**: From a Mohegan-Pequot word meaning "long tidal river."
#    - **Massachusetts**: Named after the Massachusett tribe; the name means "at or about the great hill."

# 2. **Spanish**: Some state names come from Spanish explorers and settlers.
#    - **California**: Possibly named after a mythical island in a Spanish novel, "Las sergas de Esplandián" by Garci Rodríguez de Montalvo.
#    - **Colorado**: Named after the Colorado River, which in turn was named for its red-colored silt, from Spanish "colorado" meaning "colored red."
#    - **Florida**: Named by Spanish explorer Juan Ponce de León in 1513, "Pascua Florida" refers to the "feast of the flowers" (Easter) and the verdant landscape.

# 3. **French**: Some states have names derived from French explorers and settlers or from French interpretations of Native American words.
#    - **Louisiana**: Named in honor of King Louis XIV of France.
#    - **Vermont**: From the French "vert mont," meaning "green mountain."

# 4. **English**: Many states have names that are either directly taken from places in England or are English translations or adaptations of Native American names.
#    - **New York**: Named after the Duke of York (later King James II of England), when the English took it from the Dutch who called it New Amsterdam.
#    - **Virginia**: Named after Queen Elizabeth I of England, the "Virgin Queen."

# 5. **Other European Languages**: A few state names have origins in other European languages due to the influence of various European settlers and explorers.
#    - **Illinois**: French adaptation of an Algonquin word meaning "tribe of superior men."
#    - **Minnesota**: From the Dakota Sioux word "mnisota," meaning "cloudy water" or "sky-tinted water."

# 6. **Latin or Greek**: Some names are derived from Latin or Greek words or roots.
#    - **Pennsylvania**: Named by William Penn after his father and means "Penn's woods" in Latin.

# 7. **Place Names and Royalty**: Some states are named after monarchs or other places.
#    - **Georgia**: Named in honor of King George II of Great Britain.
#    - **Maryland**: Named in honor of Queen Henrietta Maria, the wife of King Charles I of England.

# 8. **Combination of Languages and Influences**: Some state names are a blend of influences from different languages or are adaptations that have transformed over time due to usage and pronunciation.
#    - **Arkansas**: Named after the French plural of a Quapaw word "akakaze" meaning "land of downriver people."

# 9. **Geographical Features**: Some states are named after prominent geographical features.
#    - **Indiana**: Meaning "land of the Indians."
#    - **Tennessee**: Likely derived from a Cherokee village name "Tanasi."

# Keep in mind this is not an exhaustive list, and many of the origins can be more complex, with different theories about their etymology.
#   Dataset/category: helpful_base

# Example 3:
#   Prompt: Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don't take advantage of me?
#   Baseline model: gpt4_1106_preview
#   Baseline model response: Kickball is a fun and simple game that is similar to baseball or softball, but instead of hitting a ball with a bat, players kick a rubber ball. Here's how the game is typically played:

# **Equipment and Setup:**
# - A large rubber ball (similar to a playground ball).
# - A field with four bases arranged like a diamond (home plate, first base, second base, third base).
# - Teams can vary in size, but 8 to 11 players per team is common.

# **Objective:**
# The objective of kickball is to score more runs than the opposing team by kicking the ball and running around the bases to reach home plate.

# **Basic Rules:**
# 1. **Teams:** There are two teams – the kicking team and the fielding team. The teams switch roles after each inning.

# 2. **Kicking Order:** The kicking team has a lineup that determines the order in which players kick the ball.

# 3. **Pitching:** The pitcher from the fielding team rolls the ball towards home plate. The ball must be rolled smoothly (no bouncing) and within reach of the kicker.

# 4. **Kicking:** The kicker attempts to kick the ball and then run to at least first base. The kicker must kick the ball within the kicking box (an area including and surrounding home plate).

# 5. **Outs:** A player can be put out in several ways, similar to baseball:
#    - **Catch:** If the fielding team catches the ball in the air after it has been kicked (before it touches the ground), the kicker is out.
#    - **Tag:** If a fielder tags a runner with the ball while the runner is not on a base, the runner is out.
#    - **Force Out:** If a fielder has control of the ball and touches the base that a runner is forced to run to before the runner gets there (similar to a force out in baseball), the runner is out.
#    - **Strikeouts:** A kicker can also strike out if they miss the ball three times or if they kick the ball foul on the third strike.

# 6. **Running the Bases:** After kicking the ball, the runner attempts to safely reach a base. Runners can be tagged out by the fielding team or forced out when the fielding team tags the base ahead of the runner.

# 7. **Scoring Runs:** A run is scored when a runner successfully kicks the ball and then touches all four bases in order, ending at home plate.

# 8. **Innings:** An inning consists of both teams having a turn to kick and field. Each team's turn continues until they get three outs, after which the other team gets to kick.

# 9. **Foul Balls:** A kick that lands in foul territory (outside the first or third-base lines) is considered a foul ball. Two fouls count as a strike, and a third foul may result in a strikeout, depending on the rules you are playing with.

# 10. **Game Length:** The game typically lasts for a set number of innings (often five or seven) or until a time limit is reached.

# Remember that kickball is often played casually, and the rules can vary depending on the group you're playing with. Make sure to clarify any specific rules with your sister and her friends before you start playing. Most importantly, have fun and enjoy the game!
#   Dataset/category: helpful_base

# Example 4:
#   Prompt: What is some cool music from the 1920s?
#   Baseline model: gpt4_1106_preview
#   Baseline model response: The 1920s, often referred to as the "Roaring Twenties," was a period that saw a significant explosion in the popularity of jazz and the beginning of widespread commercial recording. Here are some influential artists and songs from that era:

# 1. **Louis Armstrong**: Known as "Satchmo" or "Pops," Armstrong was one of the most influential figures in jazz music. His innovative approach to playing the trumpet and his scat singing style were highly influential. Notable recordings include "West End Blues" and "Heebie Jeebies."

# 2. **Bessie Smith**: Dubbed the "Empress of the Blues," Smith was the most popular female blues singer of the decade. Her powerful voice and emotive style brought her great fame. Songs like "Downhearted Blues" and "St. Louis Blues" were among her biggest hits.

# 3. **Duke Ellington**: A pivotal bandleader and composer in jazz, Ellington led his orchestra at the Cotton Club in Harlem. Some of his notable compositions from the 1920s include "Black and Tan Fantasy" and "Mood Indigo."

# 4. **Jelly Roll Morton**: A pianist and composer, Morton claimed to be the inventor of jazz. Whether or not that's true, his influence on the genre is undeniable. His recordings with the Red Hot Peppers, like "King Porter Stomp" and "Black Bottom Stomp," are classics.

# 5. **George Gershwin**: Gershwin was a composer who bridged the gap between classical music and jazz. His 1924 composition "Rhapsody in Blue" is one of the most famous pieces of American music from the 20th century.

# 6. **Paul Whiteman**: Known as the "King of Jazz," Whiteman led one of the most popular dance bands of the 1920s. His orchestra's recording of Gershwin's "Rhapsody in Blue" with Gershwin at the piano was a huge hit.

# 7. **The Original Dixieland Jass Band**: Often credited with making the first jazz recordings in 1917, they continued to be popular through the 1920s with songs like "Dixie Jass Band One-Step" and "Tiger Rag."

# 8. **Ma Rainey**: Another significant blues singer of the era, known as the "Mother of the Blues," her powerful voice and charismatic performances made her a key figure of the blues. Songs like "See See Rider" and "Ma Rainey's Black Bottom" were popular.

# 9. **Fletcher Henderson**: As a bandleader and arranger, Henderson was influential in the development of big band jazz. His band was one of the top African American bands of the 1920s.

# 10. **Ethel Waters**: She was a blues, jazz, and gospel vocalist who became one of the first African American women to be recorded. "Stormy Weather" is one of her most enduring hits, although it was released just slightly later, in 1933.

# This list is just a starting point, as the 1920s were a rich period for music, with various styles emerging and evolving. The era's sound was characterized by the advent of new recording technologies, the rise of radio broadcasting, and the speakeasy culture during Prohibition in the United States.
#   Dataset/category: helpful_base

# Example 5:
#   Prompt: How do I wrap a present neatly?
#   Baseline model: gpt4_1106_preview
#   Baseline model response: Wrapping a present neatly can be quite straightforward with a little practice. Here's a step-by-step guide:

# ### Materials Needed:
# - Wrapping paper
# - Scissors
# - Clear tape
# - The gift to be wrapped
# - Ribbon or bow (optional)
# - Gift tag (optional)

# ### Instructions:

# 1. **Measure the Paper**:
#    - Place the gift on the wrapping paper to measure how much you'll need. You want enough paper to cover the item completely with a little overlap.
#    - Ensure that the paper is long enough to go around the gift with about 2 inches (5 cm) of overlap.
#    - Make sure there's enough paper to cover the ends of the present with extra to fold and tape.

# 2. **Cut the Paper**:
#    - Cut the wrapping paper along a straight line. You can use the grid lines on the back of the paper as a guide if available.

# 3. **Center the Gift**:
#    - Place your gift face down on the wrapping paper. This way, the seam of the paper will be on the bottom of the present.

# 4. **Wrap the Paper Around the Gift**:
#    - Bring the paper around the gift so that it's completely covered.
#    - Pull the paper taut, but be careful not to tear it.
#    - Tape one edge of the paper to the box, then pull the other edge over it to create a seam along the center of the bottom of the box. Tape it down.

# 5. **Fold the Ends**:
#    - At one end of the box, push the sides of the paper inward, creating two flaps and a triangle at the top and bottom.
#    - Fold down the top triangular flap, creasing well, then fold up the bottom triangle. This should create a neat edge.
#    - Secure with tape. Repeat on the other end.

# 6. **Final Touches**:
#    - If there are any untidy edges or corners, you can fold them neatly and secure them with a small piece of tape.
#    - Add a ribbon or bow on top for decoration, and attach a gift tag if needed.

# ### Tips for a Neat Wrap:
# - Use double-sided tape to avoid visible tape lines for a cleaner look.
# - Keep your folds sharp by creasing them with your fingers or the edge of a ruler.
# - If your wrapping paper is wrinkled, you can lightly iron it on a low setting to smooth it out (make sure the paper is iron-safe).
# - Practice makes perfect. Your first few attempts might not be perfect, but you'll improve with each gift you wrap.

# By following these steps, you should be able to neatly wrap your present and give it that professional-looking touch.
#   Dataset/category: helpful_base

python -m tests.functional.test_alpaca2