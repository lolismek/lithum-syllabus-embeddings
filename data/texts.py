"""
Ancient texts dataset with chronological metadata.
Contains 9 foundational texts from ancient literature.
"""

texts = [
    {
        "title": "Exaltation of Inanna",
        "description": "An ancient Sumerian hymn composed by Enheduanna, the high priestess of the moon god Nanna. This work celebrates the goddess Inanna, depicting her as a powerful deity of love, war, and political authority. The hymn combines personal devotion with cosmic theology, establishing the goddess's supremacy over other deities. It represents one of the earliest known examples of authored literature, blending liturgical poetry with political ideology in ancient Mesopotamia.",
        "chronological_order": 1,
        "approx_date": -2300
    },
    {
        "title": "Epic of Gilgamesh",
        "description": "A Mesopotamian epic poem following King Gilgamesh of Uruk and his companion Enkidu as they undertake heroic quests. The narrative explores themes of friendship, the quest for immortality, and humanity's relationship with the divine. After Enkidu's death, Gilgamesh journeys to discover the secret of eternal life, ultimately learning to accept human mortality. The epic includes the story of a great flood, prefiguring later flood narratives, and examines the boundaries between civilization and wilderness, human and divine.",
        "chronological_order": 2,
        "approx_date": -2100
    },
    {
        "title": "Genesis",
        "description": "The first book of the Hebrew Bible, presenting creation narratives, genealogies, and the foundational stories of the Israelite people. It begins with God creating the world in six days, followed by accounts of Adam and Eve, Cain and Abel, Noah's flood, and the Tower of Babel. The patriarchal narratives follow Abraham, Isaac, Jacob, and Joseph, tracing the covenant between God and the chosen people. Genesis establishes monotheistic theology, divine providence, and the moral framework that underpins Judeo-Christian tradition.",
        "chronological_order": 3,
        "approx_date": -1400
    },
    {
        "title": "Iliad",
        "description": "Homer's epic poem set during the final year of the Trojan War, focusing on the wrath of Achilles and its consequences. The narrative explores heroic honor, divine intervention, and the tragic costs of warfare. The conflict between Achilles and Agamemnon drives the plot, as Greek and Trojan heroes battle for glory while gods manipulate events from Olympus. The work examines themes of mortality, fate, and the warrior code, depicting both the glory and futility of war through vivid battle scenes and profound human moments.",
        "chronological_order": 4,
        "approx_date": -750
    },
    {
        "title": "Odyssey",
        "description": "Homer's epic poem recounting Odysseus's ten-year journey home to Ithaca after the Trojan War. The hero faces mythical creatures, divine obstacles, and supernatural challenges including the Cyclops, Circe, the Sirens, and Scylla and Charybdis. Parallel to his travels, the narrative follows his son Telemachus's coming of age and his wife Penelope's faithful resistance to suitors. The epic explores themes of cunning intelligence, nostalgia, identity, and the struggle to return home, both physically and spiritually, after prolonged absence and transformation.",
        "chronological_order": 5,
        "approx_date": -725
    },
    {
        "title": "Sappho Fragments",
        "description": "Lyric poems by Sappho of Lesbos, one of ancient Greece's most celebrated poets, known for intense personal expression and emotional depth. Her work explores themes of love, desire, beauty, and longing, often addressing women in her community. Written in vivid, sensory language, the fragments capture intimate moments and powerful emotions with economy and precision. Though most of her work survives only in fragments, her influence on Western lyric poetry has been profound, establishing models for personal voice and erotic expression that resonate across millennia.",
        "chronological_order": 6,
        "approx_date": -600
    },
    {
        "title": "Oresteia",
        "description": "Aeschylus's tragic trilogy comprising Agamemnon, The Libation Bearers, and The Eumenides, tracing the curse on the House of Atreus. The cycle begins with Agamemnon's return from Troy and his murder by his wife Clytemnestra, continuing through their son Orestes's revenge matricide, and concluding with his trial and acquittal in Athens. The trilogy explores justice, vengeance, the transition from blood feud to legal process, and the role of divine will in human affairs. It represents the height of Greek tragic theater and philosophical exploration of moral law.",
        "chronological_order": 7,
        "approx_date": -458
    },
    {
        "title": "Symposium",
        "description": "Plato's philosophical dialogue set at a drinking party where Socrates and other Athenian intellectuals deliver speeches in praise of Eros, the god of love. Each speaker offers a different perspective on love's nature, ascending from physical desire to philosophical contemplation. Socrates recounts the teachings of Diotima, a priestess who describes love as the desire for beauty and immortality, culminating in the vision of absolute Beauty itself. The dialogue explores the relationship between earthly and divine love, the nature of desire, and the role of beauty in philosophical ascent. It represents a masterpiece of Platonic philosophy, blending dramatic characterization with metaphysical speculation.",
        "chronological_order": 8,
        "approx_date": -385
    },
    {
        "title": "Aeneid",
        "description": "Virgil's epic poem narrating the journey of Aeneas, a Trojan hero, from the ruins of Troy to the founding of Rome. The work combines adventure, romance, and political ideology, linking Roman imperial destiny to mythical origins. Aeneas's journey includes a tragic love affair with Dido of Carthage, a descent to the underworld, and wars in Italy to establish a new homeland. The epic synthesizes Greek literary tradition with Roman values of duty, piety, and sacrifice, creating a national mythology that legitimizes Augustus's regime and celebrates Roman civilization.",
        "chronological_order": 9,
        "approx_date": -19
    },
    {
        "title": "Gospel of Mark",
        "description": "The earliest Christian Gospel, presenting a swift-paced narrative of Jesus's ministry, death, and resurrection. Mark portrays Jesus as the suffering Son of God who performs miracles, teaches in parables, and challenges religious authorities. The Gospel emphasizes discipleship, the messianic secret, and the paradox of a crucified messiah. Its urgent, dramatic style moves rapidly from Galilee to Jerusalem, culminating in the passion narrative. The work establishes the Gospel genre, combining biographical elements with theological proclamation, and shapes Christian understanding of Jesus's identity and mission.",
        "chronological_order": 10,
        "approx_date": 70
    }
]


def get_texts():
    """Return the list of texts."""
    return texts


def get_text_by_index(index):
    """Get a specific text by index."""
    return texts[index]


def get_chronological_order():
    """Return texts in chronological order."""
    return sorted(texts, key=lambda x: x['approx_date'])
