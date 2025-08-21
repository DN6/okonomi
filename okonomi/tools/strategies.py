import json
import os

from mcp.server.fastmcp import FastMCP


os.environ["MCP_TIMEOUT"] = "1000000"
mcp = FastMCP("creative_strategies_tool")


def blend_conceptual_spaces() -> str:
    return """
    Conceptual Blending Strategy:

    Merge two distinct conceptual domains to create novel ideas by identifying
    structural correspondences and projecting elements into a new blended space.
    This strategy excels at generating breakthrough innovations through unexpected
    combinations.

    Core Process:
    1. Select two conceptual domains that appear unrelated
    2. Map the deep structure and organizing principles of each domain
    3. Identify potential correspondences and analogical mappings
    4. Selectively project elements from both spaces into a blended space
    5. Develop emergent properties unique to the blend

    Key Principles:
    - Focus on structural similarities, not surface features
    - Look for organizing principles and relational patterns
    - Allow contradictions to generate creative tension
    - Develop what emerges naturally from the blend
    """


def replacement_template() -> str:
    return """
    Replacement Template Strategy:

    Use symbolic substitution to gain fresh perspectives by replacing conventional
    elements with unexpected alternatives. This forces you to see familiar concepts
    through new lenses and discover hidden assumptions.

    Core Process:
    1. Identify a key trait or function you want to highlight
    2. Find symbols, objects, or concepts strongly associated with that trait
    3. Create a context where that trait is essential for success
    4. Replace the conventional element with your subject
    5. Explore how this substitution reveals new insights

    Key Principles:
    - Focus on essential functions rather than surface appearances
    - Use strong symbolic associations to your advantage
    - Create contexts that amplify the desired trait
    - Look for what the replacement reveals about both elements

    """


def forced_connections() -> str:
    return """
    Forced Connections Strategy:

    Deliberately bridge your creative challenge with randomly selected or
    seemingly unrelated domains. The constraint of making "impossible"
    connections often reveals unexpected insights and breakthrough solutions.

    Core Process:
    1. Clearly define your creative challenge or problem space
    2. Select a random or distant domain (nature, sports, cooking, etc.)
    3. Map key elements, principles, and functions from the random domain
    4. Force connections by asking "How is this like that?"
    5. Develop practical applications from the most promising connections

    Key Principles:
    - Embrace the constraint - the more distant, the better
    - Look for transferable principles, not literal applications
    - Use analogical reasoning to bridge the gap
    - Focus on functional similarities and adaptive strategies

    Best for: Breaking creative blocks, finding fresh angles, systematic innovation
    Examples: What can urban planning learn from ant colonies? How is marketing like gardening?
    """


def assumption_reversal() -> str:
    return """
    Assumption Reversal Strategy:

    Challenge fundamental premises and taken-for-granted assumptions to reveal
    new possibilities. By systematically questioning what "everyone knows,"
    you often discover revolutionary approaches hiding in plain sight.

    Core Process:
    1. List all fundamental assumptions embedded in your challenge
    2. Identify the most taken-for-granted or "obvious" assumptions
    3. Systematically reverse or eliminate each assumption
    4. Explore what becomes possible in this assumption-free space
    5. Develop practical ideas that leverage these reversals

    Key Principles:
    - Question assumptions at multiple levels (cultural, industry, personal)
    - Look for assumptions so basic they're invisible
    - Consider both reversing and completely eliminating assumptions
    - Build solutions that thrive without the reversed assumptions

    """


def distance_associations() -> str:
    return """
    Distance Associations Strategy:

    Explore semantic relationship networks by systematically moving away from
    obvious associations toward more distant but still meaningful connections.
    This creates a structured path to unexpected insights.

    Core Process:
    1. Establish your anchor concept as the starting point
    2. Generate immediate associations (close semantic distance)
    3. Move to medium-distance concepts that require one logical step
    4. Explore far-distance concepts that maintain tenuous but real connections
    5. Create meaningful bridges between distant concepts and your challenge

    Distance Levels:
    - Close: Direct attributes, functions, or contexts
    - Medium: Related domains, analogous situations, indirect connections
    - Far: Abstract principles, emotional associations, metaphorical links

    Key Principles:
    - Maintain some thread of connection, no matter how distant
    - Use each distance level to reveal different types of insights
    - Look for surprising patterns that emerge across distance levels

    """


def perspective_shifting() -> str:
    return """
    Perspective Shifting Strategy:

    Systematically view your challenge through different viewpoints, roles,
    timeframes, or scales to reveal aspects invisible from your default
    perspective. Each shift illuminates different facets of the problem space.

    Core Process:
    1. Identify your current default perspective on the challenge
    2. Choose alternative viewpoints (stakeholder, temporal, scale, cultural)
    3. Fully inhabit each new perspective, noting what becomes visible
    4. Look for conflicts and contradictions between perspectives
    5. Synthesize insights that honor multiple viewpoints

    Perspective Types:
    - Stakeholder: Different users, opponents, beneficiaries
    - Temporal: Past, future, different time scales
    - Scale: Zoom in (micro) or out (macro, systems level)
    - Cultural: Different backgrounds, values, worldviews
    - Functional: Different disciplines or domains of expertise
    """


# Strategy registry
STRATEGIES = {
    "blend_conceptual_spaces": blend_conceptual_spaces,
    "replacement_template": replacement_template,
    "forced_connections": forced_connections,
    "assumption_reversal": assumption_reversal,
    "distance_associations": distance_associations,
    "perspective_shifting": perspective_shifting,
}


@mcp.tool()
def fetch_creative_strategy(strategy_name: str) -> str:
    """Fetch instructions for a creative strategy.

    Args:
        strategy_name: Name of the strategy to fetch.

    Returns:
        JSON with strategy_name and instructions.
    """
    if strategy_name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        return f"Unknown strategy '{strategy_name}'. Available strategies: {available}"

    instructions = STRATEGIES[strategy_name]()
    return json.dumps({"instructions": instructions})


@mcp.tool()
def list_creative_strategies() -> str:
    """List all available creative strategies with descriptions and examples.

    Returns:
        Formatted text with strategy names, descriptions, use cases, and examples.
    """
    strategy_info = {
        "blend_conceptual_spaces": {
            "description": "Merge disparate conceptual domains for breakthrough innovations",
            "best_for": "Breakthrough innovations, cross-domain insights, metaphorical thinking",
            "examples": "Digital + Physical spaces, Music + Architecture, Cooking + Software",
        },
        "replacement_template": {
            "description": "Use symbolic substitution to reveal hidden assumptions",
            "best_for": "Reframing problems, highlighting hidden qualities, challenging assumptions",
            "examples": "If a library were a restaurant, If teamwork were cooking, If data were water",
        },
        "forced_connections": {
            "description": "Bridge unrelated domains to find unexpected insights",
            "best_for": "Breaking creative blocks, finding fresh angles, systematic innovation",
            "examples": "What can urban planning learn from ant colonies? How is marketing like gardening?",
        },
        "assumption_reversal": {
            "description": "Challenge fundamental premises for paradigm shifts",
            "best_for": "Paradigm shifts, disruptive innovation, challenging status quo",
            "examples": "What if customers designed the product? What if failure was celebrated?",
        },
        "distance_associations": {
            "description": "Explore semantic networks from close to far connections",
            "best_for": "Systematic exploration, uncovering hidden connections, expanding perspective",
            "examples": "From 'smartphone' → communication → relationships → intimacy → vulnerability",
        },
        "perspective_shifting": {
            "description": "View challenges through different lenses and viewpoints",
            "best_for": "Understanding complexity, finding blind spots, building empathy",
            "examples": "How would a child/expert/alien view this? What about in 100 years?",
        },
    }

    result = "Available Creative Strategies:\n\n"
    for strategy, info in strategy_info.items():
        result += f"• Name: {strategy}\n"
        result += f"  Description: {info['description']}\n"
        result += f"  Best for: {info['best_for']}\n"
        result += f"  Examples: {info['examples']}\n\n"

    result += "Use fetch_creative_strategy(strategy_name) to get detailed guidance for any strategy."
    return result


if __name__ == "__main__":
    mcp.run()
