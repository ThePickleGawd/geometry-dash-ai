#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include "tcpserver/server.hpp"

using namespace geode::prelude;

$on_mod(Loaded)
{
	log::info("Mod loaded, let's setup tcp server");
	tcpserver::start();
}

class $modify(MyPlayerObject, PlayerObject)
{
	bool pushButton(PlayerButton btn)
	{
		log::info("Jump button received by player!");
		// PlayerObject::lockPlayer();
		this->m_isLocked = !this->m_isLocked;
		return PlayerObject::pushButton(btn);
	}
};

class $modify(MyPlayLayer, PlayLayer)
{
	bool init(GJGameLevel *level, bool p1, bool p2)
	{
		log::info("Level started");

		// TODO: Pause initialize tcp server

		return PlayLayer::init(level, p1, p2);
	}

	void destroyPlayer(PlayerObject *player, GameObject *p1)
	{
		PlayLayer::destroyPlayer(player, p1);

		if (!player->m_isDead)
			return;

		log::info("player died at {} percent", (m_player1->getPositionX() / m_levelLength) * 100.0f); // there's probably a field for this but idk what it is
	}
};
