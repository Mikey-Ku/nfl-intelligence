"""The "On Field Players" list, from text boxes as the screen reader returns them."""

from dataclasses import dataclass

from nfl_intel.personnel import Listed, OnField, name_players, parse_on_field


@dataclass
class Box:
    text: str
    box: tuple
    confidence: float = 1.0


def _at(x, y, text):
    return Box(text, (x, y - 6, x + 8 * len(text), y + 6))


def _screen():
    """What the reader sees: page furniture, the header, and the list in three columns."""
    out = [_at(158, 57, "pro.nfl.com/film/plays?season=2026"), _at(763, 276, "Q4 08:21"), _at(767, 291, "2ND & 7"),
           _at(35, 479, "Play Situation"), _at(1231, 276, "ON FIELD PLAYERS"), _at(1270, 316, "Falcons Defense")]
    defense = [("#92", "DT", "Maason Smith"), ("#96", "DE", "Zach Harrison"), ("#22", "OLB", "Yasir Abdullah"), ("#48", "OLB", "Bralen Trice"),
               ("#00", "ILB", "Divine Deablo"), ("#56", "ILB", "Harold Perkins Jr."), ("#21", "CB", "Mike Hughes"), ("#26", "CB", "C.J. Henderson"),
               ("#33", "CB", "Billy Bowman Jr."), ("#03", "SS", "Jessie Bates"), ("#31", "FS", "Xavier Watts")]
    offense = [("#10", "OB", "Jordan Love"), ("#30", "RB", "Chris Brooks"), ("#00", "WR", "Matthew Golden"), ("#09", "WR", "Christian Watson"),
               ("#16", "WR", "Bo Melton"), ("·85", "TE", "Tucker Kraft"), ("#52", "", "Jager Burton"), ("#62", "", "Jacob Monk"),
               ("#75", "", "Sean Rhyan"), ("#77", "", "Jordan Morgan"), ("#71", "", "Anthony Belton")]
    y = 344
    for num, pos, name in defense:
        out += [_at(1231, y + 2, num), _at(1310, y, name)] + ([_at(1270, y, pos)] if pos else [])
        y += 20
    out.append(_at(1270, y + 19, "Packers Offense"))
    y += 48
    for num, pos, name in offense:
        out += [_at(1231, y + 2, num), _at(1310, y - 1, name)] + ([_at(1270, y - 2, pos)] if pos else [])
        y += 20
    out += [_at(587, 839, "Sideline"), _at(382, 888, "(8:21) (Shotgun) J.Love pass short right to M.Golden for 15")]
    return out


def test_the_list_is_read_with_its_misreads_mended():
    got = parse_on_field(_screen())
    assert got.complete and (got.offense_team, got.defense_team) == ("Packers", "Falcons")
    assert got.offense[0] == Listed(10, "QB", "Jordan Love")               # "OB" on the screen reader's first try
    assert got.offense[5] == Listed(85, "TE", "Tucker Kraft")              # "·85"
    assert [p.position for p in got.offense[6:]] == ["OL"] * 5             # single letters too small to read
    assert got.defense[5].last_name == "Perkins" and got.defense[-1].name == "Xavier Watts"
    assert got.personnel == "11 personnel: 1 RB, 1 TE, 3 WR"
    assert got.package == "Nickel, 2-4-5"


def test_no_list_on_screen():
    assert parse_on_field([_at(10, 10, "Q4 08:21"), _at(10, 40, "GB ON ATL 15")]) is None
    assert OnField().personnel == "" and OnField().package == ""


def test_names_match_the_play_by_play_spelling():
    golden = Listed(0, "WR", "Matthew Golden")
    assert golden.is_called("M.Golden") and not golden.is_called("J.Golden") and not golden.is_called("")
    assert Listed(56, "ILB", "Harold Perkins Jr.").is_called("H.Perkins")


def test_only_the_certain_get_names():
    on_field = parse_on_field(_screen())
    ball = 23.6
    # trips to the offense's left with the tight end attached, one receiver right, the back beside the quarterback
    found = [(1, -1.0, 45.0), (2, -1.5, 38.0), (3, -1.6, 29.8), (4, -4.5, 21.0), (5, -1.2, 9.0)]
    names = name_players(on_field, found, ball, target="M.Golden", pass_location="right")
    assert names == {4: "Brooks", 3: "Kraft", 5: "Golden"}
    # thrown left, where two receivers stand: nobody can say which one is Golden
    assert name_players(on_field, found, ball, target="M.Golden", pass_location="left") == {4: "Brooks", 3: "Kraft"}
    # two backs on the list: the one back we found could be either
    on_field.offense[2].position = "RB"
    assert 4 not in name_players(on_field, found, ball)
    assert name_players(None, found, ball) == {}


def test_the_last_man_standing_is_named():
    on_field = OnField(offense=[Listed(10, "QB", "Jordan Love"), Listed(30, "RB", "Chris Brooks"), Listed(0, "WR", "Matthew Golden"),
                                Listed(85, "TE", "Tucker Kraft"), Listed(88, "TE", "Luke Musgrave"), Listed(89, "TE", "Ben Sims")]
                       + [Listed(60 + i, "OL", f"Line Man{i}") for i in range(5)])
    found = [(1, -1.0, 45.0), (2, -1.5, 30.0), (3, -1.6, 17.5), (4, -4.5, 21.0), (5, -5.0, 27.5)]
    names = name_players(on_field, found, 23.6)
    assert names == {}                                       # two men in the backfield, one back listed; three tight ends


def test_an_extra_lineman_is_called_jumbo():
    heavy = OnField(offense=[Listed(9, "QB", "A B"), Listed(15, "RB", "C D"), Listed(87, "TE", "E F"), Listed(89, "TE", "G H")]
                    + [Listed(60 + i, pos, f"Line Man{i}") for i, pos in enumerate(["C", "C", "G", "G", "G", "T", "T"])],
                    defense=[Listed(90 + i, pos, f"De Fender{i}") for i, pos in enumerate(["DT", "DT", "DT", "OLB", "OLB", "ILB", "ILB", "CB", "CB", "SS", "FS"])])
    assert heavy.personnel == "Jumbo, 7 linemen: 1 RB, 2 TE, 0 WR"
    assert heavy.package == "Base, 3-4-4"
