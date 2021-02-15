# -*- coding: utf-8 -*-

"""Database models."""
from . import db


class UserInput(db.Model):
	__tablename__ = 'user_input'
	id = db.Column(db.Integer, primary_key = True)
	osoite = db.Column(db.String(50), nullable = False)
	kunta = db.Column(db.String(50), nullable = False)
	postinumero = db.Column(db.String(5), nullable = False)
	asuntotyyppi = db.Column(db.String(20), nullable = False)
	asuinala = db.Column(db.Float, nullable = False)
	rakennusvuosi = db.Column(db.Integer, nullable = False)
	huone_lkm = db.Column(db.Integer, nullable=False)
	kerros = db.Column(db.Integer, nullable=False)
	kerros_yht = db.Column(db.Integer, nullable=False)
	kunto = db.Column(db.String, nullable=False)
	tontti = db.Column(db.String, nullable = False)
	vastike = db.Column(db.Float, nullable = False)
	vuokrattu = db.Column(db.Integer, nullable=False)
	hissi = db.Column(db.Integer, nullable=False)
	sauna = db.Column(db.Integer, nullable=False)
	parveke = db.Column(db.Integer, nullable=False)
	tonttiala = db.Column(db.Float, nullable = True)
	muu_kerrosala = db.Column(db.Float, nullable = True)
	created_on = db.Column(db.DateTime, index=False, unique=False, nullable=True)
	user = db.Column(db.String, nullable = False)
	query_id = db.Column(db.String(10), nullable=False)
	hinta = db.Column(db.Numeric, nullable=True)
	lat = db.Column(db.Numeric, nullable = True)
	lng = db.Column(db.Numeric, nullable = True)


class ApiKey(db.Model):
	__tablename__ = 'api_keys'
	user_id = db.Column(db.Integer, db.ForeignKey('user_input.id'), primary_key = True, nullable = False)
	apikey = db.Column(db.String(40), nullable = False, unique = True)